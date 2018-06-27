#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/core/dispatch/Dispatcher.h"

namespace caffe2 {

namespace details {
template<size_t...> struct true_t : std::true_type {};
template<class State> inline std::unique_ptr<State> init_state() {
    return c10::guts::make_unique<State>();
}
template<> inline std::unique_ptr<void> init_state<void>() {
    return std::unique_ptr<void>();
}
}

/**
 * To make a c10 operator "C10Add" callable from caffe2 as "C2MyAddOpName", just write
 *
 *     REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(C10Add, C2MyAddOpName)
 *
 * Note: This wrapper currently only supports C10 ops that have exactly one output and take that
 *       in the last parameter as "Tensor* output".
 * TODO: Figure out a better way to handle output parameters
 */

template<class OpSchemaDef, class Context, class State>
class C10OperatorWrapper final : public Operator<Context> {
    using Schema = c10::OpSchema<OpSchemaDef>;
public:

    USE_OPERATOR_CONTEXT_FUNCTIONS;

    static constexpr bool op_has_context_argument = std::is_same<CPUContext*, c10::guts::typelist::last_t<typename Schema::signature::parameter_types>>::value;
    static constexpr bool op_has_state_argument = !std::is_same<void, State>::value;

    C10OperatorWrapper(const OperatorDef& operator_def, Workspace* ws)
            : Operator<Context>(operator_def, ws) {
        state_ = details::init_state<State>();
    }

    static constexpr size_t num_inputs() {
        return Schema::signature::num_args - 1 - (op_has_context_argument ? 1 : 0) - (op_has_state_argument ? 1 : 0);
    }

    bool RunOnDevice() override {
        RunOnDevice_(c10::guts::make_index_sequence<num_inputs()>());
        return true;
    }

private:
    template<size_t... InputIndex>
    c10::guts::enable_if_t<details::true_t<InputIndex...>::value && op_has_context_argument && op_has_state_argument, void> RunOnDevice_(c10::guts::index_sequence<InputIndex...>) {
        c10::Dispatcher<OpSchemaDef>::call(Input(InputIndex)..., Output(0), state_.get(), &context_);
    }

    template<size_t... InputIndex>
    c10::guts::enable_if_t<details::true_t<InputIndex...>::value && op_has_context_argument && !op_has_state_argument, void> RunOnDevice_(c10::guts::index_sequence<InputIndex...>) {
        c10::Dispatcher<OpSchemaDef>::call(Input(InputIndex)..., Output(0), &context_);
    }

    template<size_t... InputIndex>
    c10::guts::enable_if_t<details::true_t<InputIndex...>::value && !op_has_context_argument && op_has_state_argument, void> RunOnDevice_(c10::guts::index_sequence<InputIndex...>) {
        c10::Dispatcher<OpSchemaDef>::call(Input(InputIndex)..., Output(0), state_.get());
    }

    template<size_t... InputIndex>
    c10::guts::enable_if_t<details::true_t<InputIndex...>::value && !op_has_context_argument && !op_has_state_argument, void> RunOnDevice_(c10::guts::index_sequence<InputIndex...>) {
        c10::Dispatcher<OpSchemaDef>::call(Input(InputIndex)..., Output(0));
    }

    std::unique_ptr<State> state_;
};


CAFFE_DECLARE_REGISTRY(
    C10OperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH(OpSchemaDef, State, Name)           \
  CAFFE_REGISTER_CLASS(C10OperatorRegistry, Name, C10OperatorWrapper<OpSchemaDef, CPUContext, State>)

}
