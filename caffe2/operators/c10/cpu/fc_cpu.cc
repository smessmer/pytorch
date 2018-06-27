#include "caffe2/operators/c10/schemas/fc.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::CPUContext;

namespace {
template<class DataType>
void fc_op_cpu_impl(Tensor<CPUContext> X, Tensor<CPUContext> W, Tensor<CPUContext> b, Tensor<CPUContext>* Y, caffe2::ops::FullyConnected::Cache* cache, CPUContext* context) {

    // TODO these are supposed to be arguments, fixing them for now
    constexpr size_t axis_ = 1;
    constexpr size_t axis_w_ = 1;
    constexpr bool TransposeWeight = true; // TODO This was a template argument, not a parameter

    CAFFE_ENFORCE(b.ndim() == 1, b.ndim());
    // batch size
    const auto canonical_axis = X.canonical_axis_index(axis_);
    const auto M = X.size_to_dim(canonical_axis);
    const auto K = X.size_from_dim(canonical_axis);
    const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
    const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                  : W.size_from_dim(canonical_axis_w);

    auto dimErrorString = [&]() {
        return caffe2::MakeString(
                "Dimension mismatch: ",
                "X: ",
                X.dims(),
                ", W: ",
                W.dims(),
                ", b: ",
                b.dims(),
                ", axis: ",
                axis_,
                ", M: ",
                M,
                ", N: ",
                N,
                ", K: ",
                K);
    };

    // Error checking
    CAFFE_ENFORCE(M == X.size() / K, dimErrorString());
    CAFFE_ENFORCE(K == W.size() / N, dimErrorString());
    CAFFE_ENFORCE(N == b.dim32(0), dimErrorString());
    CAFFE_ENFORCE(N == b.size(), dimErrorString());

    cache->Y_shape_cache_ = X.dims();
    // This is an invariant of canonical_axis, so we can DCHECK.
    DCHECK_LE(canonical_axis + 1, cache->Y_shape_cache_.size());
    cache->Y_shape_cache_.resize(canonical_axis + 1);
    cache->Y_shape_cache_[canonical_axis] = N;
    Y->Resize(cache->Y_shape_cache_);
    CAFFE_ENFORCE(M * N == Y->size(), dimErrorString());

    if (X.size() == 0) {
        // skip the rest of the computation if X is empty
        Y->template mutable_data<DataType>();
        return;
    }

    // default to FLOAT as math.h does.
    caffe2::TensorProto::DataType math_type = caffe2::TensorProto_DataType_FLOAT;
    if (caffe2::fp16_type<DataType>()) {
        math_type = caffe2::TensorProto_DataType_FLOAT16;
    }

    // W * x
    caffe2::math::Gemm<DataType, caffe2::CPUContext, caffe2::DefaultEngine>(
            CblasNoTrans,
            TransposeWeight ? CblasTrans : CblasNoTrans,
            M,
            N,
            K,
            1,
            X.template data<DataType>(),
            W.template data<DataType>(),
            0,
            Y->template mutable_data<DataType>(),
            context,
            math_type);
    // Add bias term
    if (cache->bias_multiplier_.size() != M) {
        // If the helper bias multiplier is not M, reshape and fill it with one.
        cache->bias_multiplier_.Resize(M);
        caffe2::math::Set<DataType, caffe2::CPUContext>(
                M,
                caffe2::convert::To<float, DataType>(1),
                cache->bias_multiplier_.template mutable_data<DataType>(),
                context);
    }
    caffe2::math::Gemm<DataType, caffe2::CPUContext, caffe2::DefaultEngine>(
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            1,
            1,
            cache->bias_multiplier_.template data<DataType>(),
            b.template data<DataType>(),
            1,
            Y->template mutable_data<DataType>(),
            context,
            math_type);

}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::FullyConnected)
        .kernel(&fc_op_cpu_impl<float>)
        .dispatchKey(c10::DispatchKey<3>{
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()},
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()},
                c10::details::TensorParameterDispatchKey{DeviceTypeId::CPU, LayoutId(0), caffe2::TypeMeta::Id<float>()}
        });
} // namespace c10
