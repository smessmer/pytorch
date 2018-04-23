#include <string>
#include <vector>
#include <iostream>
#include <array>

using ArgType = int;

template<class... Args> struct count final {
  static constexpr size_t value = std::tuple_size<std::tuple<Args...>>::value;
};

struct Argument final {
  size_t index;
  ArgType value;
};

class Parameter final {
private:
  size_t index_;

public:
  explicit constexpr Parameter(size_t index): index_(index) {}

  constexpr size_t index() const {
    return index_;
  }

  constexpr Argument operator=(ArgType value) const {
    return Argument{index_, value};
  }
};

class Arguments final {
public:
  template<size_t N>
  Arguments(std::array<ArgType, N> arguments): args_(arguments.begin(), arguments.end()) {}

  ArgType get(Parameter parameter) const {
    return args_[parameter.index()];
  }
private:
  std::vector<ArgType> args_;
};

class ParamsBase {
public:
  constexpr ParamsBase(): parameter_count_(0) {}

  template<class... Args>
  static constexpr std::array<ArgType, count<Args...>::value> args(Args... arguments) {
    std::array<Argument, count<Args...>::value> arguments_{arguments...};
    std::array<ArgType, count<Args...>::value> result;
    // TODO Don't build intermediate std::array, directly iterate arguments instead
    for (const Argument& arg : arguments_) {
      if (arg.index >= result.size()) {
        throw std::runtime_error("Missing arg");
      }
      result[arg.index] = arg.value;
    }
    return result;
  }

protected:
  constexpr Parameter parameter() {
    return Parameter(parameter_count_++);
  }

private:
  size_t parameter_count_;
};


constexpr struct _AddOperatorParams final : public ParamsBase {
  Parameter lhs = parameter();
  Parameter rhs = parameter();
} AddOperatorParams;



class AddOperator final {
public:
  void call(const Arguments& args) {
    std::cout << "Result is " << args.get(AddOperatorParams.lhs) + args.get(AddOperatorParams.rhs) << std::endl;
  }
};

int main() {
  AddOperator op;
  op.call(AddOperatorParams.args(AddOperatorParams.lhs = 3, AddOperatorParams.rhs = 5));

  //dispatch.call("add", AddOperator::lhs = 3, AddOperator::rhs = 4);

  //Tensor a = b.conv(ConvOp::strides = ..., ConvOp::bla = 3);

  return 0;
}
