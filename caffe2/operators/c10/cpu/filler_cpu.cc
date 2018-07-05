#include "caffe2/operators/c10/schemas/filler.h"
#include "caffe2/core/dispatch/KernelRegistration.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;
using caffe2::TensorCPU;
using caffe2::CPUContext;
using caffe2::TIndex;
using std::vector;

namespace {
void filler_init(c10::ArrayRef<const Tensor<CPUContext>*> inputs, Tensor<CPUContext>* output, const std::vector<int64_t>& shape, const std::vector<int>& extra_shape, bool input_as_shape) {
    if (inputs.size()) {
        auto real_shape = vector<TIndex>{};
        if (input_as_shape) {
            // Shape input must be in CPU context
            auto& input = *inputs[0];
            CAFFE_ENFORCE_EQ(
                    input.ndim(),
                    1,
                    "When input_as_shape is true, the input must be a 1D tensor of "
                    "data type TIndex");
            auto* shape_data = input.template data<TIndex>();
            real_shape.insert(real_shape.end(), shape_data, shape_data + input.dim32(0));
        } else {
            auto& input = *inputs[0];
            real_shape.insert(real_shape.end(), input.dims().begin(), input.dims().end());
        }
        real_shape.insert(real_shape.end(), extra_shape.begin(), extra_shape.end());
        output->Resize(real_shape);
    } else {
        output->Resize(shape);
    }
}

template<class Type>
void given_tensor_fill_op_cpu_impl(c10::ArrayRef<const Tensor<CPUContext>*> inputs, Tensor<CPUContext>* output, const std::vector<int64_t>& shape, const std::vector<int>& extra_shape, bool input_as_shape, const Tensor<CPUContext>& values, CPUContext* context) {
    filler_init(inputs, output, shape, extra_shape, input_as_shape);

    //TODO T might not be the correct type to call, since float allows others.

    DCHECK_EQ(output->size(), values.size())
            << "output size: " << output->size()
            << " given size: " << values.size();
    auto* data = output->template mutable_data<Type>();
    const Type* values_data = values.template data<Type>();
    if (output->size()) {
        context->template Copy<Type, CPUContext, CPUContext>(
                output->size(), values_data, data);
    }
}

void constant_fill_op_cpu_impl(c10::ArrayRef<const Tensor<CPUContext>*> inputs, Tensor<CPUContext>* output, const std::vector<int64_t>& shape, const std::vector<int>& extra_shape, bool input_as_shape, int dtype, float value, CPUContext* context) {
    filler_init(inputs, output, shape, extra_shape, input_as_shape);

    if (dtype != caffe2::TensorProto_DataType_FLOAT && dtype != caffe2::TensorProto_DataType_DOUBLE) {
        throw std::logic_error("Only float/double implemented currently");
    }

    auto* data = output->template mutable_data<float>();
    if (output->size()) {
        caffe2::math::Set<float, CPUContext>(output->size(), value, data, context);
    }
}

void uniform_fill_op_cpu_impl(c10::ArrayRef<const Tensor<CPUContext>*> inputs, Tensor<CPUContext>* output, const std::vector<int64_t>& shape, const std::vector<int>& extra_shape, bool input_as_shape, float min, float max, CPUContext* context) {
    filler_init(inputs, output, shape, extra_shape, input_as_shape);

    if (inputs.size() == 3) {
        CAFFE_ENFORCE_EQ(1, inputs[1]->size(), "min blob must be scalar");
        CAFFE_ENFORCE_EQ(1, inputs[2]->size(), "max blob must be scalar");
        min = *inputs[1]->template data<float>();
        max = *inputs[2]->template data<float>();
        if (min > max) {
            auto shape = output->dims();
            shape[0] = 0;
            output->Resize(shape);
            output->template mutable_data<float>();
            return;
        }
    }
    caffe2::math::RandUniform<float, CPUContext>(
            output->size(),
            min,
            max,
            output->template mutable_data<float>(),
            context);
}
} // namespace

namespace c10 {
    C10_REGISTER_KERNEL(caffe2::ops::ConstantFill)
        .kernel(&constant_fill_op_cpu_impl)
        .dispatchKey(c10::DeviceTypeId::CPU);

    C10_REGISTER_KERNEL(caffe2::ops::UniformFill)
        .kernel(&uniform_fill_op_cpu_impl)
        .dispatchKey(c10::DeviceTypeId::CPU);

    C10_REGISTER_KERNEL(caffe2::ops::GivenTensorFill<float>)
        .kernel(&given_tensor_fill_op_cpu_impl<float>)
        .dispatchKey(c10::DeviceTypeId::CPU);

    C10_REGISTER_KERNEL(caffe2::ops::GivenTensorFill<int>)
        .kernel(&given_tensor_fill_op_cpu_impl<int>)
        .dispatchKey(c10::DeviceTypeId::CPU);

    C10_REGISTER_KERNEL(caffe2::ops::GivenTensorFill<int64_t>)
        .kernel(&given_tensor_fill_op_cpu_impl<int64_t>)
        .dispatchKey(c10::DeviceTypeId::CPU);
} // namespace c10
