#include "caffe2/operators/c10/schemas/sparse_lengths_sum.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"
#include "caffe2/perfkernels/embedding_lookup.h"


using caffe2::Tensor;
using caffe2::CPUContext;
using caffe2::TIndex;

namespace {

template <typename InputType, typename IndexType>
void sparse_lengths_sum_op_cpu_impl(Tensor<CPUContext> dataInput, Tensor<CPUContext> indicesInput, Tensor<CPUContext> lengthsInput, Tensor<CPUContext>* output) {

    using T = float;
    constexpr bool USE_MEAN = false;
    constexpr bool USE_POSITIONAL_WEIGHT = false;

    CAFFE_ENFORCE_EQ(1, indicesInput.ndim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");
    const TIndex N = dataInput.dim(0);
    const int D = dataInput.size_from_dim(1);
    const TIndex M = lengthsInput.dim(0);
    const TIndex indices_size = indicesInput.size();

    auto shape = dataInput.dims();
    shape[0] = M;
    output->Resize(shape);
    T* out_data = output->template mutable_data<T>();

    const InputType* in_data = dataInput.template data<InputType>();
    const IndexType* indices = indicesInput.template data<IndexType>();
    const int* lengths = lengthsInput.template data<int>();
    const T* in_weight = nullptr;

    // delegate work to perfkernel that branches based on architecture
    caffe2::EmbeddingLookup<IndexType, InputType, T, USE_POSITIONAL_WEIGHT>(
            D,
            M,
            indices_size,
            N,
            in_data,
            indices,
            lengths,
            in_weight,
            nullptr, // scale_bias field is only used in SparseLengths8BitsRowwiseOp
            USE_MEAN,
            out_data);
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::SparseLengthsSum)
        .kernel(&sparse_lengths_sum_op_cpu_impl<float, int32_t>)
        .dispatchKey(c10::DispatchKey<3>{
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()},
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<int32_t>()},
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<int>()}
        });
} // namespace c10
