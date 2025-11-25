#include "ir.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Value.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"

namespace py = pybind11;

class FlagTreeOpBuilder : public TritonOpBuilder {};

void init_flagtree_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::class_<flagtree::DSLRegionOp>(m, "DSLRegionOp", py::module_local(),
                                    py::dynamic_attr())
      .def("get_operation", &flagtree::DSLRegionOp::getOperation)
      .def("get_body", &flagtree::DSLRegionOp::getBody, ret::reference)
      .def("dump", &flagtree::DSLRegionOp::dump);

  py::class_<flagtree::YieldOp>(m, "YieldOp", py::module_local(),
                                py::dynamic_attr())
      .def("dump", &flagtree::YieldOp::dump);

  py::class_<FlagTreeOpBuilder, TritonOpBuilder>(
      m, "FlagTreeOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_op_builder", &FlagTreeOpBuilder::getBuilder, ret::reference)
      .def("create_dsl_region_op",
           [](FlagTreeOpBuilder &self) -> flagtree::DSLRegionOp {
             return self.create<flagtree::DSLRegionOp>(mlir::TypeRange{},
                                                       mlir::ValueRange{});
           })
      .def("create_yield_op", [](FlagTreeOpBuilder &self) -> flagtree::YieldOp {
        return self.create<flagtree::YieldOp>(ValueRange{});
      });
}
