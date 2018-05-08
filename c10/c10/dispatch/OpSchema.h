#pragma once

#include "impl/DispatchKey.h"
#include <c10/guts/Metaprogramming.h>
#include <c10/Tensor.h>

namespace c10 {

namespace details {
template<class Arg> using is_tensor_arg = std::is_same<Tensor, std::remove_cv_t<std::remove_reference_t<Arg>>>;

namespace test_is_tensor_arg {
static_assert(is_tensor_arg<Tensor>::value, "");
static_assert(is_tensor_arg<const Tensor&>::value, "");
static_assert(is_tensor_arg<Tensor&&>::value, "");
static_assert(!is_tensor_arg<int>::value, "");
}

template<class... Args> auto getTensorTypeIds_(const Args&... args) {
  return guts::filter_map<TensorTypeId, is_tensor_arg>([] (const Tensor& t) { return t._to_impl()->type_id(); }, args...);
}

// TODO Test getTensorTypeIds_

template<class T, typename = void>
struct has_signature_defined : std::false_type {};
template<class T>
struct has_signature_defined<T, guts::void_t<
  typename T::Signature
>> : std::true_type {};

// TODO Test has_signature_defined

template<class T, typename = void>
struct has_parameter_names_defined : std::false_type {};
template<class T>
struct has_parameter_names_defined<T, guts::void_t<
  decltype(T::parameter_names)
>> : std::true_type {};

// TODO Test has_parameter_names_defined

template<class OpSchemaDef> class OpSignatureSchema final {
  static_assert(details::has_signature_defined<OpSchemaDef>::value, "Operator schema doesn't define a valid Signature member type.");
  static_assert(guts::is_function_type<typename OpSchemaDef::Signature>::value, "Signature member of operator schema must be a function type.");

  using signature_traits = guts::function_traits<typename OpSchemaDef::Signature>;
public:
  using func_type = typename signature_traits::func_type;
  using return_type = typename signature_traits::return_type;
  using parameter_types = typename signature_traits::parameter_types;

  static constexpr size_t num_args = parameter_types::size;
  static constexpr size_t num_tensor_args = guts::typelist::count_if<details::is_tensor_arg, parameter_types>::value;

private:
  static_assert(details::has_parameter_names_defined<OpSchemaDef>::value, "Operator schema doesn't define parameter_names member.");
  // TODO Allow simpler definition of parameter_names without having to spell out the std::array type in the schema def.
  static_assert(std::is_same<const std::array<const char*, num_args>, decltype(OpSchemaDef::parameter_names)>::value, "Operator schema defines parameter_names member, but it isn't the correct type. Must be a static constexpr std::array of const char* with one entry for each parameter.");

public:
  static constexpr const std::array<const char*, num_args>& parameter_names() {
    return OpSchemaDef::parameter_names;
  }
};

template<class T, typename = void>
struct has_function_dispatch_key_defined : std::false_type {};
template<class T>
struct has_function_dispatch_key_defined<T, guts::void_t<
  decltype(&T::dispatch_key)
>> : std::true_type {};

// General case. Operator doesn't overwrite DispatchKey generation. Use default.
template<class OpSchemaDef, class Enable = void> class OpDispatchKeySchema final {};
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, std::enable_if_t<!has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

public:
  using dispatch_key_type = DispatchKey<signature::num_tensor_args>;

  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return dispatch_key_type {
      details::getTensorTypeIds_(args...)
    };
  }
};

// Special case. Operator overwrites DispatchKey generation. Use that.
template<class OpSchemaDef>
class OpDispatchKeySchema<OpSchemaDef, std::enable_if_t<has_function_dispatch_key_defined<OpSchemaDef>::value>> final {
  using signature = OpSignatureSchema<OpSchemaDef>;

  static_assert(guts::is_function_type<decltype(OpSchemaDef::dispatch_key)>::value, "Operator schema defines dispatch_key member, but it isn't a function.");

  using dispatch_key_traits = guts::function_traits<decltype(OpSchemaDef::dispatch_key)>;

public:
  using dispatch_key_type = typename dispatch_key_traits::return_type;

private:

  static_assert(guts::is_equality_comparable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have the equality operator defined. Please define it.");
  static_assert(guts::is_hashable<dispatch_key_type>::value, "Operator schema specified custom dispatch_key() derivation function, but the returned dispatch key type doesn't have an overload for std::hash. Please define it.");

  static_assert(std::is_same<
    guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename dispatch_key_traits::parameter_types>>,
    guts::typelist::map_t<std::remove_cv_t, guts::typelist::map_t<std::remove_reference_t, typename signature::parameter_types>>
    >::value, "Operator schema defines custom dispatch_key() derivation function, but the arguments don't match the operator signature.");

public:

  template<class... Args>
  static inline dispatch_key_type dispatch_key(const Args&... args) {
    using guts::typelist::map_t;
    using guts::typelist::typelist;
    static_assert(std::is_same<
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typelist<Args...>>>,
      map_t<std::remove_cv_t, map_t<std::remove_reference_t, typename signature::parameter_types>>
      >::value, "Invalid argument types passed to OpSchema::dispatch_key()");
    return OpSchemaDef::dispatch_key(args...);
  }
};

}

template<class OpSchemaDef> class OpSchema final {
public:
  using signature = details::OpSignatureSchema<OpSchemaDef>;
  using dispatch = details::OpDispatchKeySchema<OpSchemaDef>;
};

// TODO Move to test cases
namespace test_opschema {
struct SchemaDef final {
  using Signature = bool (int, Tensor, float, Tensor, Tensor, unsigned int);
  static constexpr std::array<const char*, 6> parameter_names = {
    "1", "2", "3", "4", "5", "6"
  };
};
static_assert(6 == OpSchema<SchemaDef>::signature::num_args, "test num_dispatch_args");
static_assert(3 == OpSchema<SchemaDef>::signature::num_tensor_args, "test num_dispatch_args");
static_assert(std::is_same<bool, typename OpSchema<SchemaDef>::signature::return_type>::value, "test num_dispatch_args");
static_assert(std::is_same<guts::typelist::typelist<int, Tensor, float, Tensor, Tensor, unsigned int>, typename OpSchema<SchemaDef>::signature::parameter_types>::value, "test num_dispatch_args");

// TODO test OpSchema::dispatch stuff
}

}
