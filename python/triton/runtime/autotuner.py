from __future__ import annotations
from abc import abstractmethod

import builtins
import os
import time
import inspect
import random
import statistics
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .jit import KernelInterface
from .errors import OutOfResources
from .driver import driver


class BaseAutotuner(KernelInterface):

    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by: Dict = None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
    ):
        """
        :param prune_configs_by: a dict of functions that are used to prune configs, fields:
            'perf_model': performance model used to predicate running time with different configs, returns running time
            'top_k': number of configs to bench
            'prune_num_stages_by'(optional): a function used to prune num_stages. It takes configs:List[Config] as its input, and returns pruned configs.
        """
        if not configs:
            self.configs = [
                Config(
                    {},
                    num_warps=4,
                    num_stages=2,
                    num_ctas=1,
                    num_buffers_warp_spec=0,
                    num_consumer_groups=0,
                    reg_dec_producer=0,
                    reg_inc_consumer=0,
                )
            ]
        else:
            self.configs = configs
        self.keys = key
        self.cache = {}
        self.arg_names = arg_names

        # Reset to zero or restore values
        self.reset_to_zero = []
        if reset_to_zero is not None:
            self.reset_to_zero = list(reset_to_zero)
        self.restore_value = []
        if restore_value is not None:
            self.restore_value = list(restore_value)

        # Hook to reset or restore for required tensors
        self.pre_hook = lambda kwargs, reset_only=False: 0
        self.post_hook = lambda kwargs, exception: 0
        self.user_defined_pre_hook = False
        self.user_defined_post_hook = False
        if pre_hook:
            self.pre_hook = pre_hook
            self.user_defined_pre_hook = True
        elif len(self.reset_to_zero) > 0 or len(self.restore_value) > 0:

            def _pre_hook(kwargs, reset_only=False):
                for name in self.reset_to_zero:
                    kwargs[name].zero_()
                if not reset_only:
                    self.restore_copies = {
                        name: kwargs[name].clone() for name in self.restore_value
                    }

            self.pre_hook = _pre_hook

        if post_hook:
            self.post_hook = post_hook
            self.user_defined_post_hook = True
        elif len(self.restore_value) > 0:

            def _post_hook(kwargs, exception):
                for name in self.restore_value:
                    kwargs[name].copy_(self.restore_copies[name])
                self.restore_copies = {}

            self.post_hook = _post_hook

        self.perf_model = None
        self.configs_top_k = 1.0
        self.early_config_prune = None
        if prune_configs_by:
            self.perf_model = prune_configs_by.get("perf_model", self.perf_model)
            self.configs_top_k = prune_configs_by.get("top_k", self.configs_top_k)
            self.early_config_prune = prune_configs_by.get(
                "early_config_prune", self.early_config_prune
            )

        self.fn = fn
        self.base_fn = fn
        while not inspect.isfunction(self.base_fn):
            self.base_fn = self.base_fn.fn

        self.num_warmups = warmup
        self.num_reps = rep
        self.use_cuda_graph = use_cuda_graph

        # If we got explicitly called via the old interface, raise a warning
        # and proceed with the old behavior.
        if warmup is not None or rep is not None or use_cuda_graph:
            import warnings

            warnings.warn(
                (
                    "warmup, rep, and use_cuda_graph parameters are deprecated. See "
                    "https://github.com/triton-lang/triton/pull/4496 for details."
                ),
                DeprecationWarning,
                stacklevel=1,
            )
            if use_cuda_graph:
                from ..testing import do_bench_cudagraph

                self.do_bench = lambda kernel_call, quantiles: do_bench_cudagraph(
                    kernel_call,
                    rep=rep if rep is not None else 100,
                    quantiles=quantiles,
                )
                return

            import triton.testing

            self.do_bench = lambda kernel_call, quantiles: triton.testing.do_bench(
                kernel_call,
                warmup=warmup if warmup is not None else 25,
                rep=rep if rep is not None else 100,
                quantiles=quantiles,
            )
            return

        if do_bench is None:
            self.do_bench = driver.active.get_benchmarker()
        else:
            self.do_bench = do_bench

    def _bench(self, *args, config, **meta):
        from ..compiler.errors import CompileTimeAssertionFailure

        # check for conflicts, i.e. meta-parameters both provided
        # as kwargs and by the autotuner
        conflicts = meta.keys() & config.kwargs.keys()
        if conflicts:
            raise ValueError(
                f"Conflicting meta-parameters: {', '.join(conflicts)}."
                " Make sure that you don't re-define auto-tuned symbols."
            )
        # augment meta-parameters with tunable ones
        current = dict(meta, **config.all_kwargs())
        full_nargs = {**self.nargs, **current}

        def kernel_call():
            if config.pre_hook:
                config.pre_hook(full_nargs)
            self.pre_hook(full_nargs)
            try:
                self.fn.run(
                    *args,
                    **current,
                )
            except Exception as e:
                try:
                    self.post_hook(full_nargs, exception=e)
                finally:
                    # Throw exception raised by `self.fn.run`
                    raise

            self.post_hook(full_nargs, exception=None)

        try:
            return self.do_bench(kernel_call, quantiles=(0.5, 0.2, 0.8))
        except (OutOfResources, CompileTimeAssertionFailure):
            return [float("inf"), float("inf"), float("inf")]

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError(
            f"run() not implemented for {self.__class__.__name__}."
        )

    def prune_configs(self, kwargs):
        pruned_configs = self.configs
        if self.early_config_prune:
            pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
        if self.perf_model:
            top_k = self.configs_top_k
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(self.configs) * top_k)
            if len(pruned_configs) > top_k:
                est_timing = {
                    config: self.perf_model(
                        **self.nargs,
                        **kwargs,
                        **config.all_kwargs(),
                    )
                    for config in pruned_configs
                }
                pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[
                    :top_k
                ]
        return pruned_configs

    def warmup(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        ret = []
        for config in self.prune_configs(kwargs):
            ret.append(
                self.fn.warmup(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
                )
            )
        self.nargs = None
        return ret

    def _get_key(self, keys: Dict[str, Any]) -> Tuple[Any]:
        args = {k: v for (k, v) in keys.items() if k in self.arg_names}
        key = [args[key] for key in self.keys if key in args]
        for _, arg in args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        return tuple(key)


class Autotuner(BaseAutotuner):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )

    def run(self, *args, **kwargs):
        self.nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        if len(self.configs) > 1:
            key = self._get_key({**self.nargs, **kwargs})
            if key not in self.cache:
                # prune configs
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                bench_start = time.time()
                timings = {
                    config: self._bench(*args, config=config, **kwargs)
                    for config in pruned_configs
                }
                bench_end = time.time()
                self.bench_time = bench_end - bench_start
                self.cache[key] = builtins.min(timings, key=timings.get)
                full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)
                self.configs_timings = timings
            config = self.cache[key]
        else:
            config = self.configs[0]
        self.best_config = config
        if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1" and not used_cached_result:
            print(
                f"Triton autotuning for function {self.base_fn.__name__} finished after "
                f"{self.bench_time:.2f}s; best config selected: {self.best_config};"
            )
        if config.pre_hook is not None:
            full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
            config.pre_hook(full_nargs)
        ret = self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )
        self.nargs = None
        return ret


class StepwiseAutotuner(BaseAutotuner):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        min_try: int = 20,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )
        self._min_try: int = min_try
        self._tcache: Dict[
            Tuple[Any], Optional[Union[Config, Dict[Config, List[int]]]]
        ] = defaultdict(lambda: defaultdict(list))

    def run(self, *args, **kwargs):
        self.nargs: Dict[str, Any] = dict(zip(self.arg_names, args))
        key: Tuple[Any] = self._get_key({**self.nargs, **kwargs})
        cache: Optional[Union[Config, Dict[Config, List[int]]]] = self._tcache[key]
        ret = None
        while ret == None:
            isconfig: bool = isinstance(cache, Config)
            if isconfig:
                config: Config = cache
            else:
                configs = [
                    config
                    for config in self.prune_configs(kwargs)
                    if cache[config] is not None
                    and (config not in cache or len(cache[config]) < self._min_try)
                ]
                if configs:
                    config: Config = random.choice(configs)
                else:
                    config = min(
                        (k for k, v in cache.items() if v is not None),
                        key=lambda c: sum(cache[c]) / len(cache[c]),
                    )
                    self._tcache[key] = config
                    isconfig = True
            if config.pre_hook is not None:
                full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
                config.pre_hook(full_nargs)
            if not isconfig:
                di = driver.active.get_device_interface()
                start_event = di.Event(enable_timing=True)
                end_event = di.Event(enable_timing=True)
                start_event.record()
            try:
                ret = self.fn.run(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
                )
            except OutOfResources:
                if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                    args_display: List[Tuple[str, Any]] = [
                        (k, self.nargs[k]) for k in self.keys
                    ]
                    print(
                        f"Triton autotuning for function `{self.base_fn.__name__}` failed on config `{config}` with args `{args_display}`"
                    )
                ret = None
            if not isconfig:
                end_event.record()
                di.synchronize()
                timecost: float = end_event.elapsed_time(start_event)
                if ret:
                    self._tcache[key][config].append(timecost)
                else:
                    self._tcache[key][config] = None
        self.nargs = None
        return ret


class EpsilonAutotuner(BaseAutotuner):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        epsilon: float = 1.0,
        decay: float = 0.001,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )
        self._epsilon: float = epsilon
        self._decay: float = decay
        self._tcache: Dict[Tuple[Any], Tuple[Config, float, float]] = {}

    def run(self, *args, **kwargs):
        self.nargs: Dict[str, Any] = dict(zip(self.arg_names, args))
        key: Tuple[Any] = self._get_key({**self.nargs, **kwargs})
        ret = None
        while ret == None:
            if key in self._tcache:
                candidate, epsilon, perf = self._tcache[key]
                if random.random() < epsilon:
                    is_explore: bool = True
                else:
                    is_explore: bool = False
            else:
                is_explore: bool = True
                candidate: Optional[Config] = None
                epsilon: float = self._epsilon
                perf: float = sys.float_info.max
            config: Optional[Config] = None
            if is_explore:
                configs = [
                    config
                    for config in self.prune_configs(kwargs)
                    if config is not candidate
                ]
                if configs:
                    config: Config = random.choice(configs)
            if config is None:
                config: Config = candidate
            if config.pre_hook is not None:
                full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
                config.pre_hook(full_nargs)
            if is_explore:
                di = driver.active.get_device_interface()
                start_event = di.Event(enable_timing=True)
                end_event = di.Event(enable_timing=True)
                start_event.record()
            try:
                ret = self.fn.run(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
                )
            except OutOfResources:
                if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                    args_display: List[Tuple[str, Any]] = [
                        (k, self.nargs[k]) for k in self.keys
                    ]
                    print(
                        f"Triton autotuning for function `{self.base_fn.__name__}` failed on config `{config}` with args `{args_display}`"
                    )
                ret = None
            if ret is not None:
                if is_explore:
                    end_event.record()
                    di.synchronize()
                    timecost: float = end_event.elapsed_time(start_event)
                    if perf > timecost:
                        candidate = config
                        epsilon = self._epsilon
                        perf = timecost
                    else:
                        epsilon = epsilon * (1 - self._decay)
                    self._tcache[key] = (candidate, epsilon, perf)
                return ret


class ConfidenceAutotuner(BaseAutotuner):
    def __init__(
        self,
        fn,
        arg_names,
        configs,
        key,
        reset_to_zero,
        restore_value,
        pre_hook=None,
        post_hook=None,
        prune_configs_by=None,
        warmup=None,
        rep=None,
        use_cuda_graph=False,
        do_bench=None,
        ratio: float = 3.0,
    ):
        super().__init__(
            fn,
            arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            pre_hook,
            post_hook,
            prune_configs_by,
            warmup,
            rep,
            use_cuda_graph,
            do_bench,
        )
        self._ratio: float = ratio
        self._tcache: Dict[
            Tuple[Any], Optional[Union[Config, Dict[Config, List[int]]]]
        ] = defaultdict(lambda: defaultdict(list))

    def run(self, *args, **kwargs):
        self.nargs: Dict[str, Any] = dict(zip(self.arg_names, args))
        key: Tuple[Any] = self._get_key({**self.nargs, **kwargs})
        cache: Optional[Union[Config, Dict[Config, List[int]]]] = self._tcache[key]
        ret = None
        while ret == None:
            isconfig: bool = isinstance(cache, Config)
            if isconfig:
                config: Config = cache
            else:
                configs = [
                    config
                    for config in self.prune_configs(kwargs)
                    if cache[config] is not None
                ]

                def _get_boundary(
                    timelist: List[int], op: Callable[[float, float], float]
                ) -> float:
                    if timelist:
                        mean: float = statistics.mean(timelist)
                    else:
                        mean: float = sys.float_info.max
                    if len(timelist) > 1:
                        variance: float = statistics.variance(timelist)
                    elif len(timelist) == 1:
                        variance: float = sys.float_info.max
                    else:
                        variance: float = 0.0
                    return op(mean, self._ratio * variance)

                def _get_upper_boundary(timelist: List[int]) -> float:
                    return _get_boundary(timelist, lambda x, y: x + y)

                def _get_lower_boundary(timelist: List[int]) -> float:
                    return _get_boundary(timelist, lambda x, y: x - y)

                config: Config = builtins.min(
                    configs,
                    key=lambda c: _get_lower_boundary(cache[c]),
                )
                config_upper_boundary: float = _get_upper_boundary(cache[config])
                if all(
                    _get_lower_boundary(v) >= config_upper_boundary
                    for k, v in cache.items()
                    if k != config
                ):
                    self._tcache[key] = config
                    isconfig = True
            if config.pre_hook is not None:
                full_nargs = {**self.nargs, **kwargs, **config.all_kwargs()}
                config.pre_hook(full_nargs)
            if not isconfig:
                di = driver.active.get_device_interface()
                start_event = di.Event(enable_timing=True)
                end_event = di.Event(enable_timing=True)
                start_event.record()
            try:
                ret = self.fn.run(
                    *args,
                    **kwargs,
                    **config.all_kwargs(),
                )
            except OutOfResources:
                if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                    args_display: List[Tuple[str, Any]] = [
                        (k, self.nargs[k]) for k in self.keys
                    ]
                    print(
                        f"Triton autotuning for function `{self.base_fn.__name__}` failed on config `{config}` with args `{args_display}`"
                    )
                ret = None
            if not isconfig:
                end_event.record()
                di.synchronize()
                timecost: float = end_event.elapsed_time(start_event)
                if ret:
                    self._tcache[key][config].append(timecost)
                else:
                    self._tcache[key][config] = None
        self.nargs = None
        return ret


class Config:
    """
    An object that represents a possible kernel configuration for the auto-tuner to try.

    :ivar kwargs: a dictionary of meta-parameters to pass to the kernel as keyword arguments.
    :type kwargs: dict[Str, Any]
    :ivar num_warps: the number of warps to use for the kernel when compiled for GPUs. For example, if
                      `num_warps=8`, then each kernel instance will be automatically parallelized to
                      cooperatively execute using `8 * 32 = 256` threads.
    :type num_warps: int
    :ivar num_stages: the number of stages that the compiler should use when software-pipelining loops.
                       Mostly useful for matrix multiplication workloads on SM80+ GPUs.
    :type num_ctas: int
    :ivar num_ctas: number of blocks in a block cluster. SM90+ only.
    :type maxnreg: Optional[int]
    :ivar maxnreg: maximum number of registers one thread can use.  Corresponds
                       to ptx .maxnreg directive.  Not supported on all platforms.
    :ivar pre_hook: a function that will be called before the kernel is called. Parameters of this
                    function are args.
    """

    def __init__(
        self,
        kwargs,
        num_warps=4,
        num_stages=2,
        num_ctas=1,
        num_buffers_warp_spec=0,
        num_consumer_groups=0,
        reg_dec_producer=0,
        reg_inc_consumer=0,
        maxnreg=None,
        pre_hook=None,
    ):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_ctas = num_ctas
        self.num_stages = num_stages
        self.num_buffers_warp_spec = num_buffers_warp_spec
        self.num_consumer_groups = num_consumer_groups
        self.reg_dec_producer = reg_dec_producer
        self.reg_inc_consumer = reg_inc_consumer
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

    def all_kwargs(self):
        return {
            **self.kwargs,
            **{
                k: v
                for (k, v) in (
                    ("num_warps", self.num_warps),
                    ("num_ctas", self.num_ctas),
                    ("num_stages", self.num_stages),
                    ("num_buffers_warp_spec", self.num_buffers_warp_spec),
                    ("num_consumer_groups", self.num_consumer_groups),
                    ("reg_dec_producer", self.reg_dec_producer),
                    ("reg_inc_consumer", self.reg_inc_consumer),
                    ("maxnreg", self.maxnreg),
                )
                if v is not None
            },
        }

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}: {v}")
        res.append(f"num_warps: {self.num_warps}")
        res.append(f"num_ctas: {self.num_ctas}")
        res.append(f"num_stages: {self.num_stages}")
        res.append(f"num_buffers_warp_spec: {self.num_buffers_warp_spec}")
        res.append(f"num_consumer_groups: {self.num_consumer_groups}")
        res.append(f"reg_dec_producer: {self.reg_dec_producer}")
        res.append(f"reg_inc_consumer: {self.reg_inc_consumer}")
        res.append(f"maxnreg: {self.maxnreg}")
        return ", ".join(res)


def autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
):
    """
    Decorator for auto-tuning a :code:`triton.jit`'d function.

    .. highlight:: python
    .. code-block:: python

        @triton.autotune(configs=[
            triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
            triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
          ],
          key=['x_size'] # the two above configs will be evaluated anytime
                         # the value of x_size changes
        )
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE']
    :note: When all the configurations are evaluated, the kernel will run multiple times.
           This means that whatever value the kernel updates will be updated multiple times.
           To avoid this undesired behavior, you can use the `reset_to_zero` argument, which
           resets the value of the provided tensor to `zero` before running any configuration.

    If the environment variable :code:`TRITON_PRINT_AUTOTUNING` is set to
    :code:`"1"`, Triton will print a message to stdout after autotuning each
    kernel, including the time spent autotuning and the best configuration.

    :param configs: a list of :code:`triton.Config` objects
    :type configs: list[triton.Config]
    :param key: a list of argument names whose change in value will trigger the evaluation of all provided configs.
    :type key: list[str]
    :param prune_configs_by: a dict of functions that are used to prune configs, fields:
        'perf_model': performance model used to predicate running time with different configs, returns running time
        'top_k': number of configs to bench
        'early_config_prune'(optional): a function used to do early prune (eg, num_stages). It takes configs:List[Config] as its input, and returns pruned configs.
    :param reset_to_zero: a list of argument names whose value will be reset to zero before evaluating any configs.
    :type reset_to_zero: list[str]
    :param restore_value: a list of argument names whose value will be restored after evaluating any configs.
    :type restore_value: list[str]
    :param pre_hook: a function that will be called before the kernel is called.
        This overrides the default pre_hook used for 'reset_to_zero' and 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'reset_only': a boolean indicating whether the pre_hook is called to reset the values only, without a corresponding post_hook.
    :type pre_hook: lambda args, reset_only
    :param post_hook: a function that will be called after the kernel is called.
        This overrides the default post_hook used for 'restore_value'.
        'kwargs': a dict of all arguments passed to the kernel.
        'exception': the exception raised by the kernel in case of a compilation or runtime error.
    :type post_hook: lambda args, exception
    :param warmup: warmup time (in ms) to pass to benchmarking (deprecated).
    :type warmup: int
    :param rep: repetition time (in ms) to pass to benchmarking (deprecated).
    :type rep: int
    :param do_bench: a benchmark function to measure the time of each run.
    :type do_bench: lambda fn, quantiles
    """

    def decorator(fn, autotuner: str = "default"):
        autotune_dispatch: Dict[str, Callable] = {
            "default": Autotuner,
            "stepwise": StepwiseAutotuner,
            "epsilon": EpsilonAutotuner,
            "confidence": ConfidenceAutotuner,
        }
        autotuner: Callable = autotune_dispatch.get(autotuner)
        if autotuner:
            return autotuner(
                fn,
                fn.arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                pre_hook,
                post_hook,
                prune_configs_by,
                warmup,
                rep,
                use_cuda_graph,
            )
        else:
            raise NotImplementedError(f"Autotuner {autotuner} not implemented.")

    return decorator


class Heuristics(KernelInterface):

    def __init__(self, fn, arg_names, values) -> None:
        self.fn = fn
        self.values = values
        self.arg_names = arg_names

    def run(self, *args, **kwargs):
        for v, heur in self.values.items():
            kwargs[v] = heur({**dict(zip(self.arg_names, args)), **kwargs})
        return self.fn.run(*args, **kwargs)


def heuristics(values):
    """
    Decorator for specifying how the values of certain meta-parameters may be computed.
    This is useful for cases where auto-tuning is prohibitevely expensive, or just not applicable.

    .. highlight:: python
    .. code-block:: python

        @triton.heuristics(values={'BLOCK_SIZE': lambda args: 2 ** int(math.ceil(math.log2(args[1])))})
        @triton.jit
        def kernel(x_ptr, x_size, **META):
            BLOCK_SIZE = META['BLOCK_SIZE'] # smallest power-of-two >= x_size
    :param values: a dictionary of meta-parameter names and functions that compute the value of the meta-parameter.
                   each such function takes a list of positional arguments as input.
    :type values: dict[str, Callable[[list[Any]], Any]]
    """

    def decorator(fn):
        return Heuristics(fn, fn.arg_names, values)

    return decorator
