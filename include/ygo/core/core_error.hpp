#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace ygo::core {

enum class CoreErrorCode {
    Bundle,
    Callback,
    Lifecycle,
    Message,
    Response,
    Query,
};

class CoreError final : public std::runtime_error {
public:
    CoreError(CoreErrorCode code, std::string operation, std::string context)
        : std::runtime_error(operation + ": " + context), code_(code), operation_(std::move(operation)), context_(std::move(context)) {}

    CoreErrorCode code() const noexcept { return code_; }
    const std::string& operation() const noexcept { return operation_; }
    const std::string& context() const noexcept { return context_; }

private:
    CoreErrorCode code_;
    std::string operation_;
    std::string context_;
};

}  // namespace ygo::core
