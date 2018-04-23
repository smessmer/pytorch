#pragma once

#include "../TypeId.h"
#include "../ArrayRef.h"
#include "../Tensor.h"
#include "../cpu/CPUTensorImpl.h"
#include <vector>
#include <unordered_map>

namespace c10 {

/*template <class Head, class... Tail>
struct head final {
  using type = Head;
};

template <class... Args>
using head_t = typename head<Args...>::type;

template <class... Types>
struct map_tuple_types_impl;

template <class Head, class... Tail>
struct map_tuple_types_impl<Head, Tail...> final {
  template <class Result, class MapFn>
  static void call(MapFn&& mapFn, std::vector<Result>& accumulator) {
    accumulator.push_back(mapFn((Head*)nullptr));
    map_tuple_types_impl<Tail...>::call(std::forward<MapFn>(mapFn), accumulator);
  }
};

template <>
struct map_tuple_types_impl<> final {
  template <class Result, class MapFn>
  static void call(MapFn&&, std::vector<Result>&) {}
};

template <class TupleType>
struct map_tuple_types;

template <class... Types>
struct map_tuple_types<std::tuple<Types...>> final {
 private:
  template <class MapFn>
  using Result = typename std::result_of<MapFn(head_t<Types...>*)>::type;

 public:
  template <class MapFn>
  static std::vector<Result<MapFn>> call(MapFn&& mapFn) {
    std::vector<Result<MapFn>> result;
    result.reserve(std::tuple_size<std::tuple<Types...>>::value);
    map_tuple_types_impl<Types...>::call(std::forward<MapFn>(mapFn), result);
    return result;
  }
};*/

struct DispatchKey final {
  std::string name; // TODO Use constexpr-crc64
  std::vector<TypeId> arguments;
};

template<class T>
using SmallVector = ArrayRef<T>;

}

namespace std {
  template<> struct hash<c10::DispatchKey> {
    size_t operator()(c10::DispatchKey dispatchKey) const {
      size_t accumulator;
      for (TypeId id : dispatchKey.arguments) {
        accumulator += std::hash<TypeId>()(id);
        accumulator *= 2797; // prime number, reasonably large
      }
      return accumulator;
    }
  };
}

namespace c10 {

inline bool operator==(const DispatchKey& lhs, const DispatchKey& rhs) {
  return lhs.arguments == rhs.arguments;
}

class OperatorBase {
public:
  virtual void call(SmallVector<Tensor> outputs, SmallVector<Tensor> inputs) = 0;
};

class CPUAddOperator final : public OperatorBase {
public:
  void call(SmallVector<Tensor> outputs, SmallVector<Tensor> inputs) override {
    // TODO ...
  }
};

Tensor subtract(Tensor lhs, Tensor rhs) {...}
CREATE_OPERATOR(CPUSubtractOperator, &subtract);

/*
template<class T> struct is_tuple final : std::false_type {};
template<class... E> struct is_tuple<std::tuple<E...>> final : std::true_type {};

template<class T> TypeId typeId() {
  static_assert(!std::is_same<T, T>::value, "Missing typeId specialization for type");
}
template<> TypeId typeId<cpu::CPUTensorImpl>() {
  return TypeIds::CPUTensor;
}

struct getTypeId_ final {
  template<class T> TypeId operator()(T* type) {
    return typeId<typename std::remove_cv<typename std::remove_reference<decltype(*type)>::type>::type>();
  }
};
*/

template<class Inputs>
DispatchKey dispatchKey_(std::string operatorName) {
  std::vector<TypeId> inputTypes =
    map_tuple_types<Inputs>::call(getTypeId_());
  return DispatchKey{std::move(operatorName), inputTypes};
}


class Dispatcher final {
public:
  template<class Operator, class Outputs, class Inputs>
  void registerOp(std::string operatorName) {
    // TODO Check Operator signature
    // TODO make_unique
    ops_.emplace(dispatchKey_<Inputs>(std::move(operatorName)), []() -> std::unique_ptr<OperatorBase> { return std::unique_ptr<Operator>(new Operator()); });
  }

private:
  std::unordered_map<DispatchKey, std::function<std::unique_ptr<OperatorBase> ()>> ops_;
};


Tensor makeCpuTensor() {
  return Tensor::_fromImpl(nullptr); // TODO
}

void test_library() {
  Dispatcher dispatcher;
  dispatcher.registerOp<
    CPUAddOperator,
    std::tuple<cpu::CPUTensorImpl>,
    std::tuple<cpu::CPUTensorImpl, cpu::CPUTensorImpl>
  >("add");

  Tensor t1 = makeCpuTensor();
  Tensor t2 = makeCpuTensor();
  Tensor t3 = makeCpuTensor();

  dispatcher.call("add", t1, t2, t3);
}

}
