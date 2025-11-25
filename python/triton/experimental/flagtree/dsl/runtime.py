import ast
import inspect
from typing import Callable, Generic, TypeVar
from typing_extensions import override

from triton.language.core import TRITON_BUILTIN
from triton._C.libtriton import ir, flagtree_ir

T = TypeVar("T")


class CodeGenerator(ast.NodeVisitor):

    def __init__(self, builder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.builder = builder
        self.func = None

    @override
    def visit(self, module: ast.Module) -> None:
        [fn] = module.body
        self.visit_FunctionDef(fn)

    @override
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        ...


class EdslFunction(Generic[T]):

    def __init__(self, fn: T, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: T = fn

    def get_func(self, context):
        generator: CodeGenerator = CodeGenerator(context)
        src: str = inspect.getsource(self.fn)
        module: ast.Module = ast.parse(src)
        generator.visit(module)
        return generator.func


def dialect(*, name: str) -> Callable[[T], EdslFunction[T]]:

    def decorator(fn: T) -> EdslFunction[T]:
        edsl: EdslFunction = EdslFunction(fn)
        setattr(edsl, TRITON_BUILTIN, True)
        return edsl

    return decorator
