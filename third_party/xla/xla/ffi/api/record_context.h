/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_FFI_API_RECORD_CONTEXT_H_
#define XLA_FFI_API_RECORD_CONTEXT_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xla/ffi/api/api.h"
#include "xla/ffi/api/c_api.h"

namespace xla::ffi {

// Device pointers are just passed as is by pointer value.
struct DevicePointer {
  void* ptr;
};

// Host pointers are passed by value to the kernel.
struct HostPointer {
  const void* ptr;
  size_t size;
};

using KernelArg = std::variant<DevicePointer, HostPointer>;

enum class SourceFormat {
  kPtx = XLA_FFI_SourceFormat_PTX,
  kCubin = XLA_FFI_SourceFormat_CUBIN,
};

enum class RecordAction {
  kCreate = XLA_FFI_RecordAction_Create,
  kUpdate = XLA_FFI_RecordAction_Update,
};

namespace internal {

// Wrapper for accessing a pointers to opaque command pointers for recording.
// The container is capped to the maximum number of commands as specified
// during construction.
template <typename Converter>
class BoundedCommandVector {
 public:
  BoundedCommandVector(const XLA_FFI_Command** commands, size_t* num_commands,
                       size_t max_commands)
      : commands_(commands),
        num_commands_(num_commands),
        max_commands_(max_commands) {}
  // Read element by index
  const XLA_FFI_Command* operator[](size_t index) const {
    return commands_[index];
  }

  bool has_storage() const {
    return commands_ != nullptr && num_commands_ != nullptr;
  }

  auto push_back(const XLA_FFI_Command* command) {
    if (!has_storage() || *num_commands_ >= max_commands_) {
      return Converter::ResourceExhaustedError("CommandVector overflow");
    }
    commands_[(*num_commands_)++] = command;
    return Converter::Success();
  }
  size_t size() const { return *num_commands_; }
  size_t capacity() const { return max_commands_; }

 private:
  const XLA_FFI_Command** commands_;
  size_t* num_commands_;
  size_t max_commands_;
};

// Note: Cannot use absl::Overload here because it's not guaranteed that absl
// is available in the external FFI module.
template <class... Ts>
struct Overload : Ts... {
  using Ts::operator()...;
};
// Deduction guide.
template <class... Ts>
Overload(Ts...) -> Overload<Ts...>;

// Unified implementation of RecordContext for statically linked and dynamically
// linked FFI modules.
template <typename Converter>
class RecordContext {
 private:
  // Converts a span of `KernelArg` or `void*` to a vector of
  // `XLA_FFI_KernelArg`.
  template <typename KernelArgSpan>
  std::vector<XLA_FFI_KernelArg> ConvertArgs(KernelArgSpan args) {
    std::vector<XLA_FFI_KernelArg> raw_args;
    const size_t num_args = std::size(args);
    raw_args.reserve(num_args);
    for (const auto& arg : args) {
      // Can't use std/absl::remove_cvref because we don't have absl OR cpp20.
      // NOLINTNEXTLINE(modernize-type-traits)
      using ArgT = std::remove_cv_t<std::remove_reference_t<decltype(arg)>>;
      if constexpr (std::is_same_v<ArgT, KernelArg>) {
        std::visit(Overload(
                       [&](const DevicePointer& arg) {
                         raw_args.push_back(
                             {arg.ptr, 0, XLA_FFI_KernelArgType_DevicePtr});
                       },
                       [&](const HostPointer& arg) {
                         raw_args.push_back({arg.ptr, arg.size,
                                             XLA_FFI_KernelArgType_HostValue});
                       }),
                   arg);
      } else {
        // Handle void* arguments for convenience.
        raw_args.push_back({arg, 0, XLA_FFI_KernelArgType_DevicePtr});
      }
    }
    return raw_args;
  }

 public:
  using Tag = RecordCtxTag;
  using ReturnType = decltype(Converter::ToStatusOr(
      std::declval<const XLA_FFI_Command*>(), std::declval<XLA_FFI_Error*>()));

  RecordContext(XLA_FFI_RecordContext* ctx, const XLA_FFI_RecordApi* api,
                const XLA_FFI_Command** commands = nullptr,
                size_t* num_commands = nullptr, size_t max_commands = 0)
      : ctx_(ctx), api_(api), commands_(commands, num_commands, max_commands) {}

  BoundedCommandVector<Converter>& commands() { return commands_; }
  const BoundedCommandVector<Converter>& commands() const { return commands_; }

  // Unified CreateLaunch (handles both Span<const KernelArg> and Span<const
  // void* const> for args)
  template <typename KernelArgSpan,
            typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  ReturnType CreateLaunch(const char* kernel_name, const void* kernel_data,
                          size_t kernel_size, SourceFormat format,
                          XLA_FFI_LaunchDims launch_dims,
                          uint32_t shared_mem_bytes, KernelArgSpan args,
                          DepSpan dependencies = {}) {
    std::vector<XLA_FFI_KernelArg> raw_args = ConvertArgs(args);
    XLA_FFI_KernelArgs ffi_args{raw_args.data(), raw_args.size()};
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err = api_->create_launch(
        ctx_, kernel_name, kernel_data, kernel_size,
        static_cast<XLA_FFI_SourceFormat>(format), launch_dims,
        shared_mem_bytes, &ffi_args, std::data(dependencies),
        std::size(dependencies), &out_command);
    if (!err && out_command && commands_.has_storage()) {
      auto push_status = commands_.push_back(out_command);
      if (!push_status.ok()) {
        return push_status;
      }
    }
    return Converter::ToStatusOr(out_command, err);
  }

  // Unified UpdateLaunch (handles both Span<const KernelArg> and Span<const
  // void* const>)
  template <typename KernelArgSpan>
  auto UpdateLaunch(const XLA_FFI_Command* command, KernelArgSpan args) {
    std::vector<XLA_FFI_KernelArg> raw_args = ConvertArgs(args);
    XLA_FFI_KernelArgs ffi_args{raw_args.data(), raw_args.size()};
    XLA_FFI_Error* err = api_->update_launch(ctx_, command, &ffi_args);
    return Converter::ToStatus(err);
  }

  // CreateEmptyCommand with dependent commands
  template <typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  auto CreateEmptyCommand(DepSpan dependencies = {}) {
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err = api_->create_empty_command(
        ctx_, std::data(dependencies), std::size(dependencies), &out_command);
    if (!err && out_command && commands_.has_storage()) {
      auto push_status = commands_.push_back(out_command);
      if (!push_status.ok()) {
        return ReturnType(push_status);
      }
    }
    return Converter::ToStatusOr(out_command, err);
  }

  auto RequestStreamCapture() {
    XLA_FFI_Error* err = api_->request_stream_capture(ctx_);
    return Converter::ToStatus(err);
  }

  template <typename DepSpan = std::initializer_list<const XLA_FFI_Command*>>
  auto CreateMemcpyD2D(void* dst, void* src, size_t size,
                       DepSpan dependencies = {}) {
    const XLA_FFI_Command* out_command = nullptr;
    XLA_FFI_Error* err =
        api_->create_memcpy_d2d(ctx_, dst, src, size, std::data(dependencies),
                                std::size(dependencies), &out_command);
    if (!err && out_command && commands_.has_storage()) {
      auto push_status = commands_.push_back(out_command);
      if (!push_status.ok()) {
        return ReturnType(push_status);
      }
    }
    return Converter::ToStatusOr(out_command, err);
  }

  auto UpdateMemcpyD2D(const XLA_FFI_Command* command, void* dst, void* src,
                       size_t size) {
    XLA_FFI_Error* err = api_->update_memcpy_d2d(ctx_, command, dst, src, size);
    return Converter::ToStatus(err);
  }

 private:
  XLA_FFI_RecordContext* ctx_;
  const XLA_FFI_RecordApi* api_;
  BoundedCommandVector<Converter> commands_;
};

template <typename ExtensionStruct>
ExtensionStruct* FindExtension(const XLA_FFI_CallFrame* call_frame,
                               XLA_FFI_Extension_Type type) {
  if (call_frame == nullptr) {
    return nullptr;
  }
  XLA_FFI_Extension_Base* ext = call_frame->extension_start;
  while (ext != nullptr) {
    if (ext->type == type) {
      // The whole point of having tagged types is to be able to do
      // reinterpret_cast safely.
      // NOLINTNEXTLINE
      return reinterpret_cast<ExtensionStruct*>(ext);
    }
    ext = ext->next;
  }
  return nullptr;
}

template <typename T>
struct Decode<internal::CtxTag<T, RecordCtxTag>> {
  using R = T;
  static std::optional<R> call(DecodingOffsets& offsets, DecodingContext& ctx,
                               DiagnosticEngine& diagnostic) {
    auto* ext = internal::FindExtension<XLA_FFI_RecordFrame_Extension>(
        ctx.call_frame, XLA_FFI_Extension_RecordFrame);
    if (ext == nullptr || ext->record_frame == nullptr) {
      diagnostic.Emit("RecordContext is only available during RECORD stage");
      return std::nullopt;
    }
    return R(ext->record_frame->record_ctx, ext->record_frame->api,
             ext->record_frame->commands, ext->record_frame->num_commands,
             ext->record_frame->max_commands);
  }
};

template <>
struct Decode<internal::CtxTag<RecordAction>> {
  using R = RecordAction;
  static std::optional<R> call(DecodingOffsets& offsets, DecodingContext& ctx,
                               DiagnosticEngine& diagnostic) {
    auto* ext = internal::FindExtension<XLA_FFI_RecordFrame_Extension>(
        ctx.call_frame, XLA_FFI_Extension_RecordFrame);
    if (ext == nullptr || ext->record_frame == nullptr) {
      diagnostic.Emit("RecordContext is only available during RECORD stage");
      return std::nullopt;
    }
    return static_cast<RecordAction>(ext->record_frame->action);
  }
};

}  // namespace internal
}  // namespace xla::ffi

#endif  // XLA_FFI_API_RECORD_CONTEXT_H_
