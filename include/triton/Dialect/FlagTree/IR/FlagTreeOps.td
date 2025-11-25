#ifndef FLAGTREE_OPS
#define FLAGTREE_OPS

include "triton/Dialect/FlagTree/IR/FlagTreeDialect.td"
include "triton/Dialect/FlagTree/IR/FlagTreeAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class FlagTree_Op<string mnemonic, list<Trait> traits = []> :
    Op<FlagTree_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def FlagTree_DSLRegionOp : FlagTree_Op<"dsl_region", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$body);
}

def FlagTree_YieldOp : FlagTree_Op<"yield", [Pure, Terminator, ReturnLike, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Variadic<AnyType>:$outputs);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{  attr-dict ($outputs^ `:` type($outputs))? }];
}

#endif
