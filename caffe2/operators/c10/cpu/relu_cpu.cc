#include "caffe2/operators/c10/schemas/relu.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::CPUContext;

namespace {
template<class DataType>
void relu_op_cpu_impl(Tensor<CPUContext> input, Tensor<CPUContext> *output) {
    output->ResizeLike(input);

#ifdef CAFFE2_USE_ACCELERATE
    const float zero = 0.0f;
    vDSP_vthres(input.data<float>(), 1, &zero, output->mutable_data<float>(), 1, input.size());
#else
    caffe2::EigenVectorMap<float>(output->mutable_data<float>(), input.size()) =
            caffe2::ConstEigenVectorMap<float>(input.data<float>(), input.size()).cwiseMax(0.f);
#endif
    /* Naive implementation
    const float* input_data = input.data<float>();
    float* output_data = output->mutable_data<float>();
    for (int i = 0; i < input.size(); ++i) {
      output_data[i] = std::max(input_data[i], 0.f);
    }
    */
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::Relu)
        .kernel(&relu_op_cpu_impl<float>)
        .dispatchKey({DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()});
} // namespace c10
