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
#include <typeindex>
#include <unordered_map>
#include <functional>

using ArgType = int;

template<class... Args> struct count final {
  static constexpr size_t value = std::tuple_size<std::tuple<Args...>>::value;
};

// This is a workaround for getting fold-expression-like behavior in pre-C++17
template<class F, class... Ts>
F for_each(F f, Ts&&... a) {
  return (void)std::initializer_list<int>{((void)std::ref(f)(std::forward<Ts>(a)), 0)...}, f;
}

template<class T>
T* make_nullptr() {
  return nullptr;
}

template<class F, class... Ts>
F for_each(F f) {
  return for_each<F, Ts*...>(std::forward<F>(f), make_nullptr<Ts>()...);
}

template<class Tuple>
struct _for_each_tuple_element final {};
template<class... Ts>
struct _for_each_tuple_element<std::tuple<Ts...>> final {
  template<class F>
  static F call(F f) {
    return for_each<F, Ts...>(std::forward<F>(f));
  }
};
template<class Tuple, class F>
F for_each_tuple_element(F f) {
  return _for_each_tuple_element<Tuple>::call(std::forward<F>(f));
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

  template<class Param>
  typename arg_type_by_index<Param, ParamTypes...>::type operator[](Param p) const {
    return get(p);
  }
private:
  std::tuple<typename ParamTypes::value_type...> args_;
};

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
  >::value, "reorder needed");

template<class Args, class... ParamTypes>
inline constexpr Args _parse_args(Argument<ParamTypes>... args) {
  typename Args::tuple_type result;
  for_each([&result] (auto arg) {
    static_assert(decltype(arg)::index() < std::tuple_size<typename Args::tuple_type>::value, "You passed in a parameter to an op I didn't find in the op definition. Please make sure the op is defining all its parameters in its args member.");
    std::get<decltype(arg)::index()>(result) = arg.value;
  }, args...);
  return result;
}

template<class Params, class... ParamTypes>
inline constexpr typename Params::args parse_args(Argument<ParamTypes>... args) {
  return _parse_args<typename Params::args, ParamTypes...>(args...);
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

class OpId final {
public:
  constexpr OpId(uint32_t id): id_(id) {}
private:
  uint32_t id_;
  friend bool operator==(OpId lhs, OpId rhs);
  friend struct std::hash<OpId>;
};

inline bool operator==(OpId lhs, OpId rhs) {
  return lhs.id_ == rhs.id_;
}

inline bool operator!=(OpId lhs, OpId rhs) {
  return !(lhs == rhs);
}

struct DispatchKey final {
  OpId op_id;
  std::vector<std::type_index> args;
};

inline bool operator==(const DispatchKey& lhs, const DispatchKey& rhs) {
  return lhs.op_id == rhs.op_id && lhs.args == rhs.args;
}

inline bool operator!=(const DispatchKey& lhs, const DispatchKey& rhs) {
  return !(lhs==rhs);
}

namespace std {
  template<>
  struct hash<OpId> {
    // TODO constexpr hashing
    size_t operator()(OpId op_id) {
      return std::hash<uint32_t>()(op_id.id_);
    }
  };
  template<>
  struct hash<DispatchKey> {
    // TODO constexpr hashing?
    size_t operator()(const DispatchKey& key) const {
      size_t hash = std::hash<OpId>()(key.op_id);
      for (const auto& arg_type_index : key.args) {
        hash *= 7919;
        hash += std::hash<std::type_index>()(arg_type_index);
      }
      return hash;
    }
  };
}

template<class ArgsTuple>
struct genDispatchKey_ final {
  static DispatchKey call(OpId op_id) {
    // TODO Move dispatch key generation to compile time, for example inside OpDef
    std::vector<std::type_index> arg_types;
    arg_types.reserve(std::tuple_size<ArgsTuple>::value);
    for_each_tuple_element<ArgsTuple>([&arg_types] (auto* a) {
      arg_types.emplace_back(typeid(decltype(*a)));
    });
    return DispatchKey {
      .op_id = op_id,
      .args = arg_types,
    };
  }
};

template<class OpDef>
DispatchKey genDispatchKey(OpDef opDef) {
  // TODO Move dispatch key generation to compile time
  return genDispatchKey_<typename OpDef::args::tuple_type>::call(opDef.op_id);
}

class Dispatcher final {
public:
  template<class OpDef>
  void registerOp(OpDef opDef, void (*func)(typename OpDef::args)) {
    ops_.emplace(genDispatchKey(opDef), (void*)func);
  }

  template<class... ParamTypes>
  void call(OpId op_id, Argument<ParamTypes>... args) {
    using Args = typename arguments<ParamTypes...>::type;
    using FuncType = void (Args);
    // TODO Dispatch key at compile time?
    DispatchKey dispatch_key = genDispatchKey_<typename Args::tuple_type>::call(op_id);
    FuncType* func = lookup_<FuncType*>(dispatch_key);
    func(_parse_args<Args>(args...));
  }

private:
  template<class F>
  F lookup_(DispatchKey dispatch_key) {
    return (F)ops_.at(dispatch_key);
  }

  std::unordered_map<DispatchKey, void*> ops_;
};



/*
 * See below here for how operators can be implemented and called.
 * This doesn't implement any dispatching mechanism yet, but the
 * dispatcher can be built on top of this.
 */




struct MyFloat final {
    float val;
};

constexpr OpId ADD = OpId(3452);

constexpr struct {
  OpId op_id = ADD;
  // Define the parameters taken by your op here
  Parameter<0, int> lhs;
  Parameter<1, MyFloat> rhs;

  using args = arguments<decltype(lhs), decltype(rhs)>::type;
} AddOp;

class AddOperator final : public OperatorBase<AddOperator, decltype(AddOp)> {
public:
  static void doCall(args args) {
    float result = args[AddOp.lhs] + args[AddOp.rhs].val;
    printf("Result is %f\n", result);
  }
};

inline void add(decltype(AddOp)::args args) {
  float result = args[AddOp.lhs] + args[AddOp.rhs].val;
  printf("Dispatched: Result is %f\n", result);
}

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


  Dispatcher dispatch;
  dispatch.registerOp(AddOp, &add);
  dispatch.call(ADD, AddOp.lhs = 4, AddOp.rhs = MyFloat{5.6});
  dispatch.call(ADD, AddOp.rhs = MyFloat{5.6}, AddOp.lhs = 4);

  return 0;
}
