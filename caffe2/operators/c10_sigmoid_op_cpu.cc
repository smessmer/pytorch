#include "c10_sigmoid_op.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::CPUContext;

namespace {
template<class DataType>
bool sigmoid_op_cpu_impl(Tensor<CPUContext> input, Tensor<CPUContext> *output) {
    output->ResizeLike(input);

    caffe2::ConstEigenVectorArrayMap<DataType> xM(input.data<DataType>(), input.size());
    caffe2::EigenVectorArrayMap<DataType>(output->mutable_data<DataType>(), input.size()) = 1. / (1. + (-xM).exp());

    return true;
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::SigmoidOp)
        .kernel(&sigmoid_op_cpu_impl<float>)
        .dispatchKey({DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()});
} // namespace c10
