#include "caffe2/operators/c10/schemas/batch_matmul.h"
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"

using caffe2::Tensor;
using caffe2::CPUContext;

C10_DEFINE_OP_SCHEMA(caffe2::ops::BatchMatmul);

namespace {
struct TransAParameter final {
    using type = int;
    static constexpr const char* name() { return "trans_a"; }
    static constexpr int default_value() { return 0; }
};
struct TransBParameter final {
    using type = int;
    static constexpr const char* name() { return "trans_b"; }
    static constexpr int default_value() { return 0; }
};
struct BroadcastParameter final {
    using type = int;
    static constexpr const char* name() { return "broadcast"; }
    static constexpr int default_value() { return 0; }
};
struct UseScratchParameter final {
    using type = int;
    static constexpr const char* name() { return "use_scratch"; }
    static constexpr int default_value() { return 0; }
};
}

namespace caffe2 {
    REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(ops::BatchMatmul, ops::BatchMatmul::State, C10BatchMatMul_DontUseThisOpYet, TransAParameter, TransBParameter, BroadcastParameter, UseScratchParameter)
}
