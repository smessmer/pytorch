#include "caffe2/operators/c10/schemas/add.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"
#include "caffe2/operators/elementwise_ops_utils.h"

using caffe2::Tensor;
using caffe2::CPUContext;

namespace {

template<class DataType>
void add_op_cpu_impl(Tensor<CPUContext> A, Tensor<CPUContext> B, Tensor<CPUContext> *C, CPUContext* context) {

    // TODO These are supposed to be arguments
    constexpr bool legacy_broadcast_ = true;
    constexpr int axis_ = -1;

    const DataType* A_data = A.template data<DataType>();
    const DataType* B_data = B.template data<DataType>();
    std::vector<int> A_dims;
    std::vector<int> B_dims;

    if (legacy_broadcast_) {
        CAFFE_ENFORCE_NE(
                C,
                &B,
                "In-place is allowed only with the first tensor when "
                "legacy-broadcasting");
        C->ResizeLike(A);
        if (B.size() == 1) {
            A_dims = {static_cast<int>(A.size())};
            B_dims = {1};
        } else {
            size_t pre, n, post;
            std::tie(pre, n, post) =
                    caffe2::elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
            A_dims = {
                    static_cast<int>(pre), static_cast<int>(n), static_cast<int>(post)};
            B_dims = {static_cast<int>(n), 1};
        }
    } else {
        std::copy(A.dims().cbegin(), A.dims().cend(), std::back_inserter(A_dims));
        std::copy(B.dims().cbegin(), B.dims().cend(), std::back_inserter(B_dims));
        const std::vector<int> C_dims =
                caffe2::elementwise_ops_utils::ComputeBinaryBroadcastForwardDims(
                        A_dims, B_dims);
        if (C == &A) {
            CAFFE_ENFORCE_EQ(C_dims, A_dims);
        } else if (C == &B) {
            CAFFE_ENFORCE_EQ(C_dims, B_dims);
        } else {
            C->Resize(C_dims);
        }
    }
    auto* C_data =
            C->template mutable_data<DataType>();

    caffe2::math::Add(
            A_dims.size(),
            A_dims.data(),
            B_dims.size(),
            B_dims.data(),
            A.data<DataType>(),
            B.data<DataType>(),
            C->mutable_data<DataType>(),
            context);
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::Add)
        .kernel(&add_op_cpu_impl<float>)
        .dispatchKey(c10::DispatchKey<2>{
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()},
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()}
        });
} // namespace c10
