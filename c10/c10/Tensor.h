#pragma once

#include "guts/Retainable.h"
#include "ArrayRef.h"
#include "ScalarType.h"

namespace c10 { namespace guts {

  class TensorImpl;
  class UndefinedTensorImpl;

}}

namespace c10 {

// Design notes:
//  - Manual retain/release instead of shared_ptr. Reasons:
//      - PRIMARY: It's possible to work with the underlying retained object using
//        a C API, which is basically impossible to do with shared_ptr because
//        it doesn't expose a manual retain()/release() API
//      - SECONDARY: A true intrusive reference count avoids the need to store
//        a weak pointer to the control block (as is the case for
//        std::enabled_shared_from_this).
// - guts::UndefinedTensorImpl instead of null pointer. Reasons:
//      - We originally had a null pointer in ATen, but this meant that when we
//        incorrectly attempted to use such a null pointer, we would segfault and
//        crash, which is very unfriendly for our Python users.  Using an guts::UndefinedTensorImpl
//        as our default constructor is much better for us. This approach is similar to
//        allowing nullptr dispatch in Obj-C
// - Fixed the mismatch between PyTorch and C++ methods
//      - sizes() is now size()
//
// Tensor x = ...;
// Tensor y = x;  // NO COPY


// Note [Why int64_t?]
// ~~~~~~~~~~~~~~~~~~~
// We need a general purpose numeric type to represent things like sizes, strides
// and other things.  Along the way, there are a lot of hazards which you have to
// watch out for:
//
//    - size_t, the type used by most containers, is UNSIGNED, which means that
//      it is a footgun waiting to happen when you accidentally mix it up with
//      a signed quantity.
//    - int, on 64-bit systems, is still 32-bit, for backwards compatibility
//    - ssize_t is not actually signed on Windows, isn't part of the standard,
//      and only guarantees that -1 is representable
//    - long is still 32-bit on 64-bit Windows systems
//    - ptrdiff_t is allowed to have 2**15-1 as its max value in C
//
// So, we have two choices: (1) we could define our OWN integer type (typedef'ed
// to be a sane thing on all platforms), or (2) we can always use int64_t and eat
// the performance cost on 32-bit systems.  We have chosen (2).
//
// See also http://en.cppreference.com/w/cpp/language/types


// Note [Cult of the dot]
// ~~~~~~~~~~~~~~~~~~~~~
// In Python, method invocation is very simple: you write x.f()
// We wish to preserve this simplicity in C++.  To achieve this, most of our
// classes are implemented in the PIMPL pattern (there is an implementation class,
// TensorImpl, which actually contains the data and implementations of functions,
// and a wrapper class Tensor, which is just a pointer to TensorImpl), where the
// wrapper class is written to act as a pass-by-value pointer, with direct methods
// which forward to the implementation.
//
// There are a few downsides to this strategy, which we enumerate here:
//
//   - It's difficult to do const-correctness in this regime, because doing so
//     correctly requires *two* wrapper classes for the const and non-const
//     version (const Tensor doesn't cut the mustard, because it says that the
//     pointer is const, not that we have a (non-const) pointer to const data.)
//     We have opted not introduce another Tensor type, but the meaning of
//     const Tensor& is perpetually confusing to C++ experts who attempt to
//     use our Tensor type.)
//
//   - Static members that used to be pointers don't work correctly.  In particular,
//     you can't do something like this:
//
//        class Tensor {
//          static const Tensor EMPTY = {...};
//        }
//
//     At the time C++ is laying out the static member, Tensor is an incomplete type,
//     so the static Tensor EMPTY declaration is illegal.  The workaround we employ
//     in this codebase is to first predeclare a class that has all of the interesting
//     members, and then inherit from it the actual class name that now has the static
//     members (and we can implicitly convert from the pre-class.)  Not the prettiest,
//     but it gets us the desired API.


// SUMMING UP
// 1. There will NOT be a retain/release on the Tensor class.  There might be
// some unsafe mechanism to retain/release (because C API bindings need it), but it won't
// be a method with an easy name to spell.
// 2. virtual issue, DELAYED (it's OK for now)
// 3. Undefined tensor... can we make illegal states unrepresentable?
// 4. Because of ArrayRef, we will need to define code style guide (ArrayRef disagrees)

// NB: This is publically inherited, but only to conveniently bring the public methods
// of Retainable into scope.  If this is causing bad error messages, make it private
// again and explicitly 'using' each of the public methods you want to propagate.
class Tensor final {
  using TensorBase = guts::Retainable<guts::TensorImpl, guts::UndefinedTensorImpl>;
  TensorBase impl_;

  Tensor(TensorBase impl) : impl_(impl) {};

public:
  // Steals the reference.  (In old ATen, there was an optional retain which also bumped
  // the refcount while you were at it.)
  // TODO: Figure out a safer way to expose this to relevant sites
  // I've forgotten how to use this safely, so it's
  // not a good API. :)
  static Tensor _fromImpl(guts::TensorImpl* impl) { return Tensor(TensorBase(impl)); };

  // Normal constructors
  // TODO: I don't know if it's safe to replace this with = default here... godbolt time...
  Tensor()  = default;
  Tensor(const Tensor &rhs) = default;
  Tensor(Tensor &&rhs) noexcept = default;

  // These methods are SO important, they are currently implemented via virtual dispatch
  // via our implementation classes.  Most non-core methods should be implemented by
  // the generic dispatch mechanism.

  // The definitions of these live in TensorMethods.h
  // dzhulgakov: nit - is it widely used? I'd prefer ndimension as below or rank. In C2 it's a function returning particular dimension
  inline int64_t dim() const;
  // dzhulgakov: nit - why `size` and not `sizes`? In C2 the size is number of elements - I bet it will cause confusion
  inline ArrayRef<int64_t> size() const;
  inline ArrayRef<int64_t> stride() const;
  inline void* data_ptr() const;
  inline int64_t ndimension() const;
  int64_t storage_offset() const;
  int64_t numel() const;

  // dzhulgakov: what are the semantics of it? i.e. how do I change type of the elements stored in a tensor? Or is it passed only in the constructor?
  // ezyang: invocation of data() is only well-defined if the type T matches the internal type T of the tensor.
  // This function has nothing to do with casting.
  template<typename T>
  inline T *data() const;

  // The "well known" Tensor functions will call into the dispatch mechanism (yet to be
  // implemented)

  void resize_(ArrayRef<int64_t> size, ArrayRef<int64_t> stride);

  // Hmmmmm, does the void* violate our dispatch data model?  OTOH, we are probably going to
  // need ways to create tensors from void* pointers
  void copy_(ScalarType s, const void* p, int64_t size_bytes);

  // NB: This is an instance of the design pattern, where we cannot (and will not) dispatch
  // templated fucntions.  So you have to untemplate it first.
  template <typename T>
  void copy_(ArrayRef<T> arr) {
    copy_(c10::scalar_type<T>(), arr.data(), arr.size() * sizeof(T));
  }

  // To be something like:
  // Tensor add(Tensor x, Tensor y) { guts::dispatch("add", x, y); }

};

} // namespace c10
