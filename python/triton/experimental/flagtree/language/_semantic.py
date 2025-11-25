from typing import TypeVar

from triton._C.libtriton.flagtree_ir import FlagTreeOpBuilder
from triton.language.semantic import TritonSemantic

TensorTy = TypeVar("TensorTy")


class FlagTreeSemantic(TritonSemantic[TensorTy]):

    def __init__(self, builder: FlagTreeOpBuilder, *args, **kwargs) -> None:
        super().__init__(builder, *args, **kwargs)

    def call(self, func, operands):
        dsl_region_op = self.builder.create_dsl_region_op()
        pt = self.builder.get_insertion_point()
        region = dsl_region_op.get_body()
        block = self.builder.create_block_with_parent(region, [])
        self.builder.set_insertion_point_to_start(block)
        self.builder.create_yield_op()
        self.builder.restore_insertion_point(pt)
