/*
 * Ignore stuff here, there's a lot of C++ magic going on to enable this
 * and to do a lot of error checking, but that would all be hidden away in some header.
 * Also, this C++ magic happens entirely at compile time and doesn't incur any runtime cost.
 * Scroll down to the next multi-line comment to see how the API would be used.
 */

#include <string>
#include <vector>
#include <iostream>
#include <array>
#include <tuple>
#include <utility>
#include <functional>

using ArgType = int;

template<class... Args> struct count final {
  static constexpr size_t value = std::tuple_size<std::tuple<Args...>>::value;
};

// This is a workaround for getting fold-expression-like behavior in pre-C++17
template <class F, class... Ts>
F for_each(F f, Ts&&... a) {
  return (void)std::initializer_list<int>{((void)std::ref(f)(std::forward<Ts>(a)), 0)...}, f;
}

template<class ParamType>
struct Argument final {
  static constexpr size_t index() {
    return ParamType::index();
  }

  typename ParamType::value_type value;
};

template<size_t index_, class T>
class Parameter final {
public:
  using value_type = T;

  static constexpr size_t index() {
    return index_;
  }

  constexpr Argument<Parameter> operator=(T value) const {
    return Argument<Parameter>{value};
  }
};

template<class... ParamTypes>
struct param_types_incremental final {};

template<class Head1ParamType, class Head2ParamType, class... TailParamTypes>
struct param_types_incremental<Head1ParamType, Head2ParamType, TailParamTypes...> final {
  static constexpr bool value = Head1ParamType::index() + 1 == Head2ParamType::index() && param_types_incremental<Head2ParamType, TailParamTypes...>::value;
};

template<class HeadParamType>
struct param_types_incremental<HeadParamType> final {
  static constexpr bool value = true;
};

template<class HeadParamType, class... TailParamTypes>
struct param_types_valid final{
  static constexpr bool value = HeadParamType::index() == 0 && param_types_incremental<HeadParamType, TailParamTypes...>::value;
};

template<class Param, class... ParamTypes>
struct arg_type_by_index final {
  static_assert(Param::index() < count<ParamTypes...>::value, "The operator implementation tried to get a parameter that isn't defined in the op definition. Please make sure the op is defining all its parameters in its args member.");
  using type = typename std::tuple_element<Param::index(), std::tuple<ParamTypes...>>::type::value_type;
};

template<class... ParamTypes>
class Arguments final {
  static_assert(param_types_valid<ParamTypes...>::value, "Template parameters for Arguments must be ordered in sequence 0, 1, 2, ...");
public:
  Arguments(std::tuple<typename ParamTypes::value_type...> args): args_(args) {} // TODO std::forward

  using tuple_type = std::tuple<typename ParamTypes::value_type...>;

  template<class Param>
  typename arg_type_by_index<Param, ParamTypes...>::type get(Param p) const {
    return std::get<Param::index()>(args_);
  }
private:
  std::tuple<typename ParamTypes::value_type...> args_;
};
/* TODO This commented out code is an alternative to the arguments definition right below it, where the order of the parameters doesn't matter.
template<class ParamsTuple, class IndexSequence>
struct arguments_ final {};

template<class ParamsTuple, size_t... Indices>
struct arguments_<ParamsTuple, std::index_sequence<Indices...>> final {
  using type = Arguments<typename std::tuple_element<std::tuple_element<Indices, ParamsTuple>::type::index(), ParamsTuple>::type...>;
};

template<class... Params>
struct arguments final {
  using type = typename arguments_<std::tuple<Params...>, std::index_sequence_for<Params...>>::type;
};

static_assert(std::is_same<
  arguments<Parameter<0, int>, Parameter<1, std::string>>::type,
  Arguments<Parameter<0, int>, Parameter<1, std::string>>
  >::value, "no reorder needed");
static_assert(std::is_same<
  arguments<Parameter<1, std::string>, Parameter<0, int>>::type,
  Arguments<Parameter<0, int>, Parameter<1, std::string>>
  >::value, "reorder needed");
static_assert(std::is_same<
  arguments<Parameter<1, std::string>, Parameter<0, int>, Parameter<2, float>>::type,
  Arguments<Parameter<0, int>, Parameter<1, std::string>, Parameter<2, float>>
  >::value, "reorder needed");*/
template<class... Params>
struct arguments final {
  static_assert(param_types_valid<Params...>::value, "Please make sure your operator definition 'args' member definition contains all arguments and is in the correct order.");
  using type = Arguments<Params...>;
};

template<class Params, class... ParamTypes>
inline constexpr typename Params::args parse_args(Argument<ParamTypes>... args) {
  typename Params::args::tuple_type result;
  for_each([&result] (auto arg) {
    static_assert(decltype(arg)::index() < std::tuple_size<typename Params::args::tuple_type>::value, "You passed in a parameter to an op I didn't find in the op definition. Please make sure the op is defining all its parameters in its args member.");
    std::get<decltype(arg)::index()>(result) = arg.value;
  }, args...);
  return result;
}

template<class Concrete, class Params>
class OperatorBase {
public:
  template<class... ParamTypes>
  static void call(Argument<ParamTypes>... arguments) {
    Concrete::doCall(parse_args<Params>(arguments...));
  }

protected:
  using args = typename Params::args;
};







/*
 * See below here for how operators can be implemented and called.
 * This doesn't implement any dispatching mechanism yet, but the
 * dispatcher can be built on top of this.
 */




struct MyFloat final {
    float val;
};

constexpr struct {
  // Define the parameters taken by your op here
  Parameter<0, int> lhs;
  Parameter<1, MyFloat> rhs;

  using args = arguments<decltype(lhs), decltype(rhs)>::type;
} AddOp;

class AddOperator final : public OperatorBase<AddOperator, decltype(AddOp)> {
public:
  static void doCall(const args& args) {
    float result = args.get(AddOp.lhs) + args.get(AddOp.rhs).val;
    std::cout << "Result is " << result << std::endl;
  }
};

// Btw, inheriting from OperatorBase is just for syntactic sugar.
// It should be possible to implement this without an operator base class
// and operators being pure C functions, but that would need a dispatcher
// being implemented first.

int main() {
  AddOperator op;
  /*
   * Call the op with named parameters.
   * At compile time, this is transformed into a plain positional C ABI call.
   * All named parameters are gone in the binary.
   */
  op.call(AddOp.lhs = 4, AddOp.rhs = MyFloat{5.5});
  op.call(AddOp.rhs = MyFloat{5.5}, AddOp.lhs = 4);
  op.call(AddOp.lhs = 3);
  op.call(AddOp.rhs = MyFloat{5.5});
  // TODO: Can we make op.call(SomeOtherOp.arg = bla) cause a compile time error?

  return 0;
}
