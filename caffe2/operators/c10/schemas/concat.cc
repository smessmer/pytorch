
#include "caffe2/operators/c10/schemas/concat.h"
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"

using caffe2::Tensor;
using caffe2::CPUContext;

C10_DEFINE_OP_SCHEMA(caffe2::ops::Concat);

namespace {
struct AxisParameter final {
    using type = int;
    static constexpr const char* name() { return "axis"; }
    static constexpr int default_value() { return -1; }
};
struct AddAxisParameter final {
    using type = int;
    static constexpr const char* name() { return "add_axis"; }
    static constexpr int default_value() { return 0; }
};
}

namespace caffe2 {
    REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_ARRAY_INPUT_AND_PARAMETERS(ops::Concat, void, C10Concat_DontUseThisOpYet, ParameterHelper<AxisParameter>, ParameterHelper<AddAxisParameter>)
}
