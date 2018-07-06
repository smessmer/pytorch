#include "caffe2/operators/c10/schemas/flatten.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::CPUContext;

namespace {
template<class DataType>
void flatten_op_cpu_impl(const Tensor<CPUContext>& input, Tensor<CPUContext> *output, int axis, CPUContext* context) {
    CAFFE_ENFORCE_GE(
            input.dims().size(), axis, "The rank of the tensor must be >= axis.");
    output->Resize(input.size_to_dim(axis), input.size_from_dim(axis));
    context->template CopyItems<CPUContext, CPUContext>(
            input.meta(),
            input.size(),
            input.raw_data(),
            output->raw_mutable_data(input.meta()));
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::Flatten)
        .kernel(&flatten_op_cpu_impl<float>)
        .dispatchKey({DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()});
} // namespace c10
