#include "caffe2/operators/c10/schemas/relu.h"
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"

using caffe2::Tensor;
using caffe2::CPUContext;

C10_DEFINE_OP_SCHEMA(caffe2::ops::Relu);

namespace caffe2 {
    REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(ops::Relu, void, C10Relu_DontUseThisOpYet)
}
