// hlo_aot.cc
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "llvm/Support/CodeGen.h"  // For Reloc
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/platform_manager.h"  // Added for PlatformManager
#include "llvm/TargetParser/Host.h"  // For
#include "llvm/ADT/StringMap.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"  // Add for TF_ASSIGN_OR_RETURN
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"
#include "xla/error_spec.h"
#include "xla/literal_comparison.h"

using namespace xla;
using xla::HloModule;
using xla::HloComputation;
using xla::HloInstruction;

void PrintLiteralForDebug(absl::string_view label, const xla::Literal& lit) {
  std::cout << label << " (shape=" << xla::ShapeUtil::HumanString(lit.shape())
            << ")" << std::endl;
  std::cout << lit.ToStringWithoutShape() << std::endl;
}

void LogLiteralMismatch(absl::string_view context,
                        const xla::Literal& reference,
                        const xla::Literal& actual) {
  std::cout << "=== " << context << " mismatch details ===" << std::endl;
  PrintLiteralForDebug("Reference", reference);
  PrintLiteralForDebug("Actual", actual);
}

struct InstructionFragment {
  std::unique_ptr<HloModule> module;
  std::vector<const xla::HloInstruction*> produced_instructions;
  std::vector<const xla::HloInstruction*> external_operands;
  std::string description;
  int chunk_index = -1;
};

enum class BackendKind { kCpu, kGpu };

struct BackendToken {
  BackendKind kind = BackendKind::kCpu;
  std::string spec;

  std::string DebugString() const {
    switch (kind) {
      case BackendKind::kCpu:
        if (spec.empty()) {
          return "cpu<default>";
        }
        return absl::StrCat("cpu(", spec, ")");
      case BackendKind::kGpu:
        if (spec.empty()) {
          return "gpu";
        }
        return absl::StrCat("gpu(", spec, ")");
    }
    return "unknown";
  }
};

bool operator==(const BackendToken& lhs, const BackendToken& rhs) {
  return lhs.kind == rhs.kind && lhs.spec == rhs.spec;
}

// Small holder tying a fragment to its compiled executable.
struct CompiledFragment {
  InstructionFragment frag;
  std::unique_ptr<xla::PjRtLoadedExecutable> exec;
  std::string feature_set;
  int chunk_size = 0;
  BackendToken backend;
  xla::PjRtClient* client = nullptr;
  xla::PjRtDevice* device = nullptr;
};

class FeaturePolicy {
 public:
  explicit FeaturePolicy(std::vector<BackendToken> tokens)
      : tokens_(std::move(tokens)) {
    if (tokens_.empty()) {
      tokens_.push_back(BackendToken{BackendKind::kCpu, ""});
    }
    absl::flat_hash_set<std::string> seen;
    for (const auto& token : tokens_) {
      std::string key = token.DebugString();
      if (seen.insert(key).second) {
        unique_tokens_.push_back(token);
      }
    }
  }

  const BackendToken& SelectForChunk(int chunk_index) const {
    int idx = chunk_index % tokens_.size();
    if (idx < 0) {
      idx += tokens_.size();
    }
    return tokens_[idx];
  }

  const std::vector<BackendToken>& tokens() const { return tokens_; }
  const std::vector<BackendToken>& unique_tokens() const {
    return unique_tokens_;
  }
  bool IsSingleBackend() const { return unique_tokens_.size() == 1; }

 private:
  std::vector<BackendToken> tokens_;
  std::vector<BackendToken> unique_tokens_;
};

struct ParameterState {
  const xla::Literal* literal = nullptr;
  std::unique_ptr<xla::PjRtBuffer> buffer;
};

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> UploadLiteralToDevice(
    const xla::Literal& literal, xla::PjRtDevice* device) {
  if (device == nullptr) {
    return absl::InvalidArgumentError("Target device is null");
  }
  TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                      device->default_memory_space());
  TF_ASSIGN_OR_RETURN(auto buffer,
                      device->client()->BufferFromHostLiteral(literal,
                                                              memory_space));
  TF_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
  return buffer;
}

absl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> CopyBufferBetweenClients(
    xla::PjRtBuffer* source, xla::PjRtDevice* target_device) {
  if (source == nullptr) {
    return absl::InvalidArgumentError("Source buffer is null");
  }
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal,
                      source->ToLiteralSync());
  return UploadLiteralToDevice(*literal, target_device);
}

absl::StatusOr<xla::PjRtBuffer*> EnsureBufferOnDevice(
    std::unique_ptr<xla::PjRtBuffer>& buffer, xla::PjRtDevice* target_device,
    const xla::Literal* literal_fallback) {
  if (target_device == nullptr) {
    return absl::InvalidArgumentError("Target device is null");
  }
  if (buffer && buffer->device() == target_device) {
    return buffer.get();
  }
  if (buffer) {
    TF_ASSIGN_OR_RETURN(auto transferred,
                        CopyBufferBetweenClients(buffer.get(), target_device));
    buffer = std::move(transferred);
    return buffer.get();
  }
  if (literal_fallback != nullptr) {
    TF_ASSIGN_OR_RETURN(buffer, UploadLiteralToDevice(*literal_fallback,
                                                      target_device));
    return buffer.get();
  }
  return absl::InvalidArgumentError(
      "No literal fallback or buffer available for transfer");
}

absl::StatusOr<xla::PjRtDevice*> GetDefaultDevice(xla::PjRtClient* client,
                                                  absl::string_view label) {
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Requested ", label, " device but client is null"));
  }
  const auto& devices = client->devices();
  if (devices.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No devices available for ", label, " backend"));
  }
  return devices.front();
}

bool ContainsGpuFragments(
    const std::vector<CompiledFragment>& compiled_frags) {
  for (const auto& cf : compiled_frags) {
    if (cf.backend.kind == BackendKind::kGpu) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<std::vector<ParameterState>> InitializeParameterStates(
    absl::Span<const xla::Literal> literals, xla::PjRtDevice* seed_device) {
  std::vector<ParameterState> states(literals.size());
  for (int i = 0; i < literals.size(); ++i) {
    states[i].literal = &literals[i];
    TF_ASSIGN_OR_RETURN(states[i].buffer,
                        UploadLiteralToDevice(literals[i], seed_device));
  }
  return states;
}

absl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
PrepareArgumentBuffers(absl::Span<const xla::Literal> literals,
                       xla::PjRtDevice* device) {
  std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
  buffers.reserve(literals.size());
  for (const xla::Literal& literal : literals) {
    TF_ASSIGN_OR_RETURN(auto buffer, UploadLiteralToDevice(literal, device));
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

absl::Status RunFragmentsSequentiallyCpuOnly(
    const xla::HloModule& original,
    const std::vector<CompiledFragment>& compiled_frags,
    absl::Span<xla::PjRtBuffer* const> entry_param_buffers,
    std::shared_ptr<xla::Literal>* literal_out) {
  xla::ExecuteOptions exec_opts;
  exec_opts.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;
  absl::flat_hash_map<const xla::HloInstruction*,
                      std::unique_ptr<xla::PjRtBuffer>>
      value_map;
  const xla::HloComputation* entry = original.entry_computation();

  for (const auto& cf : compiled_frags) {
    if (cf.backend.kind != BackendKind::kCpu) {
      return absl::InvalidArgumentError(
          "RunFragmentsSequentiallyCpuOnly received non-CPU fragment");
    }
    const InstructionFragment& frag = cf.frag;
    std::vector<xla::PjRtBuffer*> arg_ptrs;
    arg_ptrs.reserve(frag.external_operands.size());
    for (const xla::HloInstruction* op : frag.external_operands) {
      if (op->opcode() == xla::HloOpcode::kParameter) {
        int param_idx = op->parameter_number();
        if (param_idx < 0 ||
            param_idx >= static_cast<int>(entry_param_buffers.size())) {
          return absl::InvalidArgumentError(
              absl::StrCat("Invalid parameter index ", param_idx));
        }
        arg_ptrs.push_back(entry_param_buffers[param_idx]);
      } else {
        auto it = value_map.find(op);
        if (it == value_map.end()) {
          return absl::InternalError(
              absl::StrCat("Missing value for operand ", op->name(),
                           " while executing fragment ", frag.description));
        }
        arg_ptrs.push_back(it->second.get());
      }
    }
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<xla::PjRtBuffer>> exec_outputs,
        cf.exec->ExecuteSharded(arg_ptrs, cf.device, exec_opts));
    if (exec_outputs.size() != 1) {
      return absl::InternalError(
          absl::StrCat("Expected a single output buffer from fragment ",
                       frag.description, ", got ", exec_outputs.size()));
    }
    TF_RETURN_IF_ERROR(exec_outputs[0]->GetReadyFuture().Await());
    if (frag.produced_instructions.size() != 1) {
      return absl::InternalError(
          absl::StrCat("Fragment ", frag.description,
                       " produced multiple values; chunking should prevent "
                       "this."));
    }
    value_map[frag.produced_instructions.front()] =
        std::move(exec_outputs[0]);
  }

  const xla::HloInstruction* root = entry->root_instruction();
  auto it = value_map.find(root);
  if (it == value_map.end()) {
    return absl::InternalError("No value for ROOT instruction");
  }
  if (literal_out != nullptr) {
    TF_ASSIGN_OR_RETURN(*literal_out, it->second->ToLiteralSync());
  }
  return absl::OkStatus();
}

absl::StatusOr<xla::Literal> LoadLiteralFromProtoFile(const std::string& path) {
  std::string data;
  {
    std::unique_ptr<tsl::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(
        tsl::Env::Default()->NewRandomAccessFile(path, &file));
    tsl::uint64 size = 0;
    TF_RETURN_IF_ERROR(tsl::Env::Default()->GetFileSize(path, &size));
    data.resize(size);
    tsl::StringPiece sp;
    TF_RETURN_IF_ERROR(file->Read(0, size, &sp, &data[0]));
  }

  xla::LiteralProto proto;
  if (!proto.ParseFromString(data)) {
    return absl::InternalError("Failed to parse LiteralProto from " + path);
  }
  TF_ASSIGN_OR_RETURN(xla::Literal lit, xla::Literal::CreateFromProto(proto));
  return lit;
}

absl::StatusOr<InstructionFragment> BuildFragmentFromChunk(
    const xla::HloModule& module,
    absl::Span<xla::HloInstruction* const> chunk_instructions,
    int chunk_index) {
  if (chunk_instructions.empty()) {
    return absl::InternalError("Cannot build fragment from empty chunk");
  }

  xla::HloModuleConfig cfg(module.config());
  cfg.clear_entry_computation_layout();
  auto frag = std::make_unique<xla::HloModule>(
      absl::StrCat("chunk_", chunk_index), cfg);
  xla::HloComputation::Builder builder(
      absl::StrCat("chunk_", chunk_index, "_entry"));

  absl::flat_hash_set<const xla::HloInstruction*> chunk_set(
      chunk_instructions.begin(), chunk_instructions.end());
  absl::flat_hash_map<const xla::HloInstruction*, xla::HloInstruction*>
      cloned;
  absl::flat_hash_map<const xla::HloInstruction*, xla::HloInstruction*>
      external_param_map;
  std::vector<const xla::HloInstruction*> external_operands;
  external_operands.reserve(chunk_instructions.size());
  int param_index = 0;

  for (xla::HloInstruction* instr : chunk_instructions) {
    std::vector<xla::HloInstruction*> new_operands;
    new_operands.reserve(instr->operand_count());
    for (xla::HloInstruction* operand : instr->operands()) {
      if (chunk_set.contains(operand)) {
        auto it = cloned.find(operand);
        if (it == cloned.end()) {
          return absl::InternalError(
              absl::StrCat("Operand ", operand->name(),
                           " not cloned yet while building fragment chunk ",
                           chunk_index));
        }
        new_operands.push_back(it->second);
      } else {
        auto param_it = external_param_map.find(operand);
        xla::HloInstruction* param = nullptr;
        if (param_it == external_param_map.end()) {
          param = builder.AddInstruction(xla::HloInstruction::CreateParameter(
              param_index, operand->shape(),
              absl::StrCat("param_", param_index, "_for_", instr->name())));
          external_param_map.emplace(operand, param);
          external_operands.push_back(operand);
          ++param_index;
        } else {
          param = param_it->second;
        }
        new_operands.push_back(param);
      }
    }
    xla::HloInstruction* cloned_instr =
        builder.AddInstruction(instr->CloneWithNewOperands(instr->shape(),
                                                           new_operands));
    cloned[instr] = cloned_instr;
  }

  const xla::HloInstruction* entry_root =
      module.entry_computation()->root_instruction();
  std::vector<const xla::HloInstruction*> produced_instructions;
  std::vector<xla::HloInstruction*> produced_clones;
  produced_instructions.reserve(chunk_instructions.size());
  produced_clones.reserve(chunk_instructions.size());

  for (xla::HloInstruction* instr : chunk_instructions) {
    bool needed = instr == chunk_instructions.back() || instr == entry_root;
    if (!needed) {
      for (xla::HloInstruction* user : instr->users()) {
        if (!chunk_set.contains(user)) {
          needed = true;
          break;
        }
      }
    }
    if (needed) {
      produced_instructions.push_back(instr);
      produced_clones.push_back(cloned.at(instr));
    }
  }
  if (produced_clones.empty()) {
    produced_instructions.push_back(chunk_instructions.back());
    produced_clones.push_back(cloned.at(chunk_instructions.back()));
  }

  xla::HloInstruction* root = nullptr;
  if (produced_clones.size() == 1) {
    root = produced_clones.front();
  } else {
    root = builder.AddInstruction(
        xla::HloInstruction::CreateTuple(produced_clones));
  }
  xla::HloComputation* comp = frag->AddEntryComputation(builder.Build(root));
  frag->mutable_config().SetDefaultComputationLayout(
      comp->ComputeProgramShape());

  InstructionFragment info;
  info.module = std::move(frag);
  info.produced_instructions = std::move(produced_instructions);
  info.external_operands = std::move(external_operands);
  info.description =
      absl::StrCat("chunk#", chunk_index, "_size=", chunk_instructions.size());
  info.chunk_index = chunk_index;
  return info;
}

absl::StatusOr<std::vector<InstructionFragment>>
SplitModuleWithChunkSize(const xla::HloModule& module, int chunk_size) {
  const xla::HloComputation* entry = module.entry_computation();
  std::vector<xla::HloInstruction*> order = entry->MakeInstructionPostOrder();
  std::vector<xla::HloInstruction*> worklist;
  worklist.reserve(order.size());
  for (xla::HloInstruction* instr : order) {
    if (instr->opcode() == xla::HloOpcode::kParameter) {
      continue;
    }
    worklist.push_back(instr);
  }
  if (worklist.empty()) {
    return std::vector<InstructionFragment>();
  }
  chunk_size =
      std::max(1, std::min<int>(chunk_size, static_cast<int>(worklist.size())));

  std::vector<InstructionFragment> fragments;
  fragments.reserve((worklist.size() + chunk_size - 1) / chunk_size);
  int chunk_index = 0;
  for (int i = 0; i < worklist.size();) {
    int end = std::min<int>(i + chunk_size, worklist.size());
    std::optional<InstructionFragment> selected_fragment;
    while (end > i) {
      absl::Span<xla::HloInstruction* const> attempt(worklist.data() + i,
                                                     end - i);
      TF_ASSIGN_OR_RETURN(auto fragment,
                          BuildFragmentFromChunk(module, attempt, chunk_index));
      if (fragment.produced_instructions.size() == 1) {
        selected_fragment = std::move(fragment);
        break;
      }
      --end;
    }
    if (!selected_fragment.has_value()) {
      return absl::InternalError(
          "Failed to build fragment without multiple outputs");
    }
    fragments.push_back(std::move(selected_fragment.value()));
    i = end;
    ++chunk_index;
  }
  return fragments;
}

// Compile each fragment with maybe different feature sets.
// For simplicity, this uses the same features for all; you can vary it
// by index or opcode.
absl::StatusOr<std::vector<CompiledFragment>>
CompileFragments(const std::vector<InstructionFragment>& fragments,
                 const std::string& cpu_name,
                 const FeaturePolicy& feature_policy,
                 int chunk_size, xla::PjRtCpuClient* cpu_client,
                 xla::PjRtClient* gpu_client) {

  xla::CompileOptions compile_options;

  std::vector<CompiledFragment> compiled;
  compiled.reserve(fragments.size());

  for (const auto& frag : fragments) {
    const BackendToken& backend =
        feature_policy.SelectForChunk(frag.chunk_index);
    xla::XlaComputation computation(frag.module->ToProto());
    std::unique_ptr<xla::PjRtLoadedExecutable> exec;
    xla::PjRtDevice* device = nullptr;

    if (backend.kind == BackendKind::kCpu) {
      const std::string& features = backend.spec;
      auto entry_point = std::string(frag.module->entry_computation()->name());
      auto aot_opts = xla::cpu::CpuAotCompilationOptions(
          /*triple=*/"x86_64-unknown-linux-gnu",
          /*cpu_name=*/cpu_name,
          /*features=*/features,
          /*entry_point_name=*/entry_point,
          /*relocation_model=*/
          xla::cpu::CpuAotCompilationOptions::RelocationModel::Static);

      TF_ASSIGN_OR_RETURN(
          auto compiled_exec,
          cpu_client->CompileAheadOfTimeAndLoad(computation,
                                                compile_options, aot_opts));
      exec = std::move(compiled_exec);
      TF_ASSIGN_OR_RETURN(device, GetDefaultDevice(cpu_client, "CPU"));
    } else {
      if (gpu_client == nullptr) {
        return absl::InvalidArgumentError(
            "GPU backend requested but GPU client is unavailable");
      }
      TF_ASSIGN_OR_RETURN(auto compiled_exec,
                          gpu_client->CompileAndLoad(computation,
                                                      compile_options));
      exec = std::move(compiled_exec);
      TF_ASSIGN_OR_RETURN(device, GetDefaultDevice(gpu_client, "GPU"));
    }

    CompiledFragment cf;
    InstructionFragment stored;
    stored.module =
        std::unique_ptr<xla::HloModule>(frag.module->Clone());
    stored.produced_instructions = frag.produced_instructions;
    stored.external_operands = frag.external_operands;
    stored.description = frag.description;
    cf.frag = std::move(stored);
    cf.exec = std::move(exec);
    cf.feature_set = backend.spec;
    cf.chunk_size = chunk_size;
    cf.backend = backend;
    cf.client = (backend.kind == BackendKind::kCpu)
                    ? static_cast<xla::PjRtClient*>(cpu_client)
                    : gpu_client;
    cf.device = device;
    compiled.push_back(std::move(cf));
  }

  return compiled;
}

absl::Status RunFragmentsSequentiallyMultiBackend(
    const xla::HloModule& original,
    const std::vector<CompiledFragment>& compiled_frags,
    std::vector<ParameterState>* entry_params,
    std::shared_ptr<xla::Literal>* literal_out) {
  absl::flat_hash_map<const xla::HloInstruction*,
                      std::unique_ptr<xla::PjRtBuffer>>
      value_map;
  const xla::HloComputation* entry = original.entry_computation();

  xla::ExecuteOptions exec_opts;
  exec_opts.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;

  // Execute fragments in the same order as they were generated.
  for (const auto& cf : compiled_frags) {
    const InstructionFragment& frag = cf.frag;
    xla::PjRtDevice* target_device = cf.device;

    // 1) Build the arg list for this fragment from value_map and params.
    std::vector<xla::PjRtBuffer*> arg_ptrs;
    arg_ptrs.reserve(frag.external_operands.size());

    for (const xla::HloInstruction* op : frag.external_operands) {
      if (op->opcode() == xla::HloOpcode::kParameter) {
        int param_idx = op->parameter_number();
        if (param_idx < 0 ||
            param_idx >= static_cast<int>(entry_params->size())) {
          return absl::InvalidArgumentError(
              absl::StrCat("Invalid parameter index ", param_idx));
        }
        ParameterState& state = (*entry_params)[param_idx];
        TF_ASSIGN_OR_RETURN(
            xla::PjRtBuffer * buf,
            EnsureBufferOnDevice(state.buffer, target_device, state.literal));
        arg_ptrs.push_back(buf);
      } else {
        // internal value, must have been produced by an earlier fragment
        auto it = value_map.find(op);
        if (it == value_map.end()) {
          return absl::InternalError(
              absl::StrCat("Missing value for operand ", op->name(),
                           " while executing fragment ", frag.description));
        }
        TF_ASSIGN_OR_RETURN(
            xla::PjRtBuffer * buf,
            EnsureBufferOnDevice(it->second, target_device,
                                 /*literal_fallback=*/nullptr));
        arg_ptrs.push_back(buf);
      }
    }

    // 2) Execute the fragment
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<xla::PjRtBuffer>> exec_outputs,
        cf.exec->ExecuteSharded(arg_ptrs, target_device, exec_opts));
    if (exec_outputs.size() != 1) {
      return absl::InternalError(
          absl::StrCat("Expected a single output buffer from fragment ",
                       frag.description, ", got ", exec_outputs.size()));
    }
    TF_RETURN_IF_ERROR(exec_outputs[0]->GetReadyFuture().Await());
    if (frag.produced_instructions.size() != 1) {
      return absl::InternalError(
          absl::StrCat("Fragment ", frag.description,
                       " produced multiple values; chunking should prevent "
                       "this."));
    }

    // 3) Store output in the value map
    value_map[frag.produced_instructions.front()] =
        std::move(exec_outputs[0]);
  }

  // At this point, the ROOT instruction's value is in value_map[entry->root_instruction()].
  const xla::HloInstruction* root = entry->root_instruction();
  auto it = value_map.find(root);
  if (it == value_map.end()) {
    return absl::InternalError("No value for ROOT instruction");
  }

  // Optionally convert ROOT buffer to Literal for correctness checking:
  if (literal_out != nullptr) {
    TF_ASSIGN_OR_RETURN(*literal_out, it->second->ToLiteralSync());
  }

  return absl::OkStatus();
}

struct BenchmarkStats {
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
  double ci_half_width_ms = 0.0;
  int runs = 0;
};

BenchmarkStats ComputeBenchmarkStats(const std::vector<double>& samples) {
  BenchmarkStats stats;
  if (samples.empty()) {
    return stats;
  }
  stats.runs = static_cast<int>(samples.size());
  double sum = 0.0;
  for (double v : samples) {
    sum += v;
  }
  stats.mean_ms = sum / samples.size();
  if (samples.size() > 1) {
    double variance = 0.0;
    for (double v : samples) {
      double delta = v - stats.mean_ms;
      variance += delta * delta;
    }
    variance /= (samples.size() - 1);
    stats.stddev_ms = std::sqrt(variance);
    stats.ci_half_width_ms =
        1.96 * stats.stddev_ms / std::sqrt(static_cast<double>(samples.size()));
  }
  return stats;
}

absl::StatusOr<BenchmarkStats> BenchmarkExecution(
    absl::string_view label, const std::function<absl::Status()>& run_once,
    bool skip_first_sample = false) {
  constexpr int kMaxRuns = 100;
  std::vector<double> samples;
  samples.reserve(kMaxRuns);

  if (skip_first_sample) {
    TF_RETURN_IF_ERROR(run_once());
    std::cout << label << " warmup run (discarded)" << std::endl;
  }

  for (int i = 0; i < kMaxRuns; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    TF_RETURN_IF_ERROR(run_once());
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << label << " run " << (i + 1) << ": " << elapsed_ms << " ms"
              << std::endl;
    samples.push_back(elapsed_ms);
    if (samples.size() < 2) {
      continue;
    }
    BenchmarkStats stats = ComputeBenchmarkStats(samples);
    double relative_ci =
        stats.ci_half_width_ms / std::max(stats.mean_ms, 1e-12);
    if (relative_ci <= 0.05) {
      std::cout << "Achieved 95% CI within 5% of mean for " << label
                << " after " << stats.runs << " runs." << std::endl;
      return stats;
    }
  }

  std::cout << "Reached max runs for " << label
            << ". Reporting best-effort stats." << std::endl;
  return ComputeBenchmarkStats(samples);
}

std::vector<std::vector<std::string>> ParseFeatureExperiments(
    const std::string& raw) {
  std::vector<std::vector<std::string>> experiments;
  if (raw.empty()) {
    experiments.push_back({""});
    return experiments;
  }
  for (absl::string_view exp : absl::StrSplit(raw, ';')) {
    std::string cleaned_exp = std::string(absl::StripAsciiWhitespace(exp));
    if (cleaned_exp.empty()) {
      continue;
    }
    std::vector<std::string> tokens;
    for (absl::string_view token : absl::StrSplit(cleaned_exp, '|')) {
      std::string trimmed = std::string(absl::StripAsciiWhitespace(token));
      tokens.push_back(trimmed);
    }
    if (tokens.empty()) {
      tokens.push_back("");
    }
    experiments.push_back(std::move(tokens));
  }
  if (experiments.empty()) {
    experiments.push_back({""});
  }
  return experiments;
}

std::string MaterializeFeatureString(
    const std::string& token,
    const llvm::StringMap<bool, llvm::MallocAllocator>& host_features) {
  if (token == "all") {
    std::vector<std::string> enabled;
    enabled.reserve(host_features.size());
    for (const auto& feature : host_features) {
      if (feature.getValue()) {
        enabled.push_back(feature.getKey().str());
      }
    }
    std::sort(enabled.begin(), enabled.end());
    return absl::StrJoin(enabled, ",");
  }
  return token;
}

absl::StatusOr<BackendToken> ParseBackendToken(
    const std::string& raw_token, bool enable_gpu_backend,
    const llvm::StringMap<bool, llvm::MallocAllocator>& host_features) {
  std::string cleaned = std::string(absl::StripAsciiWhitespace(raw_token));
  std::string lowered = absl::AsciiStrToLower(cleaned);
  auto payload_after_colon = [](const std::string& input) -> std::string {
    auto pos = input.find(':');
    if (pos == std::string::npos) {
      return "";
    }
    return input.substr(pos + 1);
  };

  auto is_prefix = [](absl::string_view lhs,
                      absl::string_view rhs) -> bool {
    return absl::StartsWith(lhs, rhs);
  };

  if (lowered == "gpu" || is_prefix(lowered, "gpu:")) {
    if (!enable_gpu_backend) {
      return absl::InvalidArgumentError(
          "GPU backend requested but not enabled. Pass --enable_gpu "
          "or set the enable flag when invoking the splitter.");
    }
    std::string spec = payload_after_colon(cleaned);
    return BackendToken{BackendKind::kGpu, spec};
  }

  std::string cpu_payload = cleaned;
  if (is_prefix(lowered, "cpu:")) {
    cpu_payload = payload_after_colon(cleaned);
  }
  std::string features = MaterializeFeatureString(cpu_payload, host_features);
  return BackendToken{BackendKind::kCpu, features};
}

absl::StatusOr<BenchmarkStats> BenchmarkFullExecution(
    const BackendToken& backend, const xla::HloModule& module,
    const xla::CompileOptions& compile_options,
    absl::Span<const xla::Literal> input_literals,
    xla::PjRtCpuClient* cpu_client, xla::PjRtClient* gpu_client,
    const xla::Literal& ref_lit) {
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  xla::PjRtDevice* device = nullptr;
  std::vector<std::unique_ptr<xla::PjRtBuffer>> arg_buffers;

  if (backend.kind == BackendKind::kCpu) {
    const std::string& features = backend.spec;
    auto aot_options = std::make_unique<xla::cpu::CpuAotCompilationOptions>(
        /*triple=*/"x86_64-unknown-linux-gnu", /*cpu_name=*/"sapphirerapids",
        /*features=*/features,
        /*entry_point_name=*/"main.1",
        /*relocation_model=*/
        xla::cpu::CpuAotCompilationOptions::RelocationModel::Static);

    xla::XlaComputation computation(module.ToProto());
    TF_ASSIGN_OR_RETURN(
        auto cpu_exec,
        cpu_client->CompileAheadOfTimeAndLoad(computation, compile_options,
                                              *aot_options));
    executable = std::move(cpu_exec);
    TF_ASSIGN_OR_RETURN(device, GetDefaultDevice(cpu_client, "CPU"));
    TF_ASSIGN_OR_RETURN(arg_buffers,
                        PrepareArgumentBuffers(input_literals, device));
  } else {
    if (gpu_client == nullptr) {
      return absl::InvalidArgumentError(
          "GPU backend requested but GPU client is unavailable");
    }
    xla::XlaComputation computation(module.ToProto());
    TF_ASSIGN_OR_RETURN(auto gpu_exec,
                        gpu_client->CompileAndLoad(computation,
                                                    compile_options));
    executable = std::move(gpu_exec);
    TF_ASSIGN_OR_RETURN(device, GetDefaultDevice(gpu_client, "GPU"));
    TF_ASSIGN_OR_RETURN(arg_buffers,
                        PrepareArgumentBuffers(input_literals, device));
  }

  std::vector<xla::PjRtBuffer*> arg_ptrs;
  arg_ptrs.reserve(arg_buffers.size());
  for (auto& buffer : arg_buffers) {
    arg_ptrs.push_back(buffer.get());
  }

  ExecuteOptions execute_options;
  execute_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;

  TF_ASSIGN_OR_RETURN(
      auto result_buffers,
      executable->ExecuteSharded(arg_ptrs, device, execute_options));
  TF_RETURN_IF_ERROR(result_buffers[0]->GetReadyFuture().Await());
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> full_out,
                      result_buffers[0]->ToLiteralSync());
  auto compare_status =
      literal_comparison::Near(ref_lit, *full_out, ErrorSpec(1e-1, 1e-1),
                               std::nullopt, nullptr);
  if (!compare_status.ok()) {
    std::cout << "Full execution (target=" << backend.DebugString()
              << ") output does NOT match reference!" << std::endl;
    LogLiteralMismatch(
        absl::StrCat("Full execution (target=", backend.DebugString(), ")"),
        ref_lit, *full_out);
  } else {
    std::cout << "Full execution (target=" << backend.DebugString()
              << ") output matches reference." << std::endl;
  }

  auto run_full_once = [&]() -> absl::Status {
    TF_ASSIGN_OR_RETURN(
        auto iteration_buffers,
        executable->ExecuteSharded(arg_ptrs, device, execute_options));
    TF_RETURN_IF_ERROR(iteration_buffers[0]->GetReadyFuture().Await());
    return absl::OkStatus();
  };

  return BenchmarkExecution(
      absl::StrCat("Full (target=", backend.DebugString(), ")"),
      run_full_once,
      /*skip_first_sample=*/backend.kind == BackendKind::kGpu);
}

std::string DescribePolicy(const FeaturePolicy& policy) {
  if (policy.IsSingleBackend()) {
    return policy.unique_tokens().front().DebugString();
  }
  std::vector<std::string> pieces;
  pieces.reserve(policy.tokens().size());
  for (const auto& token : policy.tokens()) {
    pieces.push_back(token.DebugString());
  }
  return absl::StrCat("cycle(", absl::StrJoin(pieces, "| "), ")");
}

absl::Status RunAotCompilationExample(std::string hlo_file,
                                      std::string features_str,
                                      std::string io_prefix,
                                      bool enable_gpu_backend) {
  xla::CompileOptions compile_options;
  llvm::StringMap<bool, llvm::MallocAllocator> host_machine_features =
      llvm::sys::getHostCPUFeatures();

  // Read HLO from file.
  std::ifstream in_file(hlo_file);
  if (!in_file) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open HLO file: ", hlo_file));
  }
  std::stringstream buffer;
  buffer << in_file.rdbuf();
  std::string hlo = buffer.str();

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(
                          hlo,
                          HloModuleConfig() /* unused */));

  xla::CpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> cpu_client_holder,
                      xla::GetXlaPjrtCpuClient(client_options));
  auto* cpu_client =
      tsl::down_cast<xla::PjRtCpuClient*>(cpu_client_holder.get());

  std::unique_ptr<xla::PjRtClient> gpu_client_holder;
  xla::PjRtClient* gpu_client = nullptr;
  if (enable_gpu_backend) {
    xla::GpuClientOptions gpu_options;
    TF_ASSIGN_OR_RETURN(gpu_client_holder,
                        xla::GetXlaPjrtGpuClient(gpu_options));
    gpu_client = gpu_client_holder.get();
  }

  int num_params = module->entry_computation_layout().parameter_count();
  std::vector<xla::Literal> input_lits;
  input_lits.reserve(num_params);
  for (int i = 0; i < num_params; ++i) {
    std::string path = absl::StrCat(io_prefix, "/input_", i, ".litpb");
    TF_ASSIGN_OR_RETURN(auto lit, LoadLiteralFromProtoFile(path));
    input_lits.push_back(std::move(lit));
  }

  TF_ASSIGN_OR_RETURN(xla::PjRtDevice * cpu_device,
                      GetDefaultDevice(cpu_client, "CPU"));
  absl::Span<const xla::Literal> input_span = absl::MakeConstSpan(input_lits);

  TF_ASSIGN_OR_RETURN(auto cpu_entry_buffer_storage,
                      PrepareArgumentBuffers(input_span, cpu_device));
  std::vector<xla::PjRtBuffer*> cpu_entry_ptrs;
  cpu_entry_ptrs.reserve(cpu_entry_buffer_storage.size());
  for (auto& buf : cpu_entry_buffer_storage) {
    cpu_entry_ptrs.push_back(buf.get());
  }

  std::string ref_path = absl::StrCat(io_prefix, "/output_0.ref.litpb");
  TF_ASSIGN_OR_RETURN(xla::Literal ref_lit,
                      LoadLiteralFromProtoFile(ref_path));

  // Determine granularity options based on non-parameter instruction count.
  int non_param_instructions = 0;
  for (xla::HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() != xla::HloOpcode::kParameter) {
      ++non_param_instructions;
    }
  }
  if (non_param_instructions == 0) {
    return absl::InvalidArgumentError(
        "Entry computation has no executable instructions");
  }
  std::vector<int> chunk_sizes;
  chunk_sizes.reserve(non_param_instructions);
  for (int size = 1; size <= non_param_instructions; ++size) {
    chunk_sizes.push_back(size);
  }

  auto feature_experiments = ParseFeatureExperiments(features_str);
  auto print_summary = [](absl::string_view label,
                          const BenchmarkStats& stats) {
    std::cout << label << " mean: " << stats.mean_ms
              << " ms (stddev " << stats.stddev_ms << " ms, 95% CI +/-"
              << stats.ci_half_width_ms << " ms over " << stats.runs << " runs)"
              << std::endl;
  };

  for (const auto& raw_tokens : feature_experiments) {
    std::vector<BackendToken> backend_tokens;
    backend_tokens.reserve(raw_tokens.size());
    for (const std::string& token : raw_tokens) {
      TF_ASSIGN_OR_RETURN(BackendToken backend_token,
                          ParseBackendToken(token, enable_gpu_backend,
                                            host_machine_features));
      backend_tokens.push_back(std::move(backend_token));
    }
    FeaturePolicy policy(std::move(backend_tokens));
    std::cout << "=== Feature policy: " << DescribePolicy(policy) << " ==="
              << std::endl;

    absl::flat_hash_map<std::string, BenchmarkStats> full_stats_map;
    for (const BackendToken& token : policy.unique_tokens()) {
      TF_ASSIGN_OR_RETURN(
          BenchmarkStats stats,
          BenchmarkFullExecution(token, *module, compile_options, input_span,
                                 cpu_client, gpu_client, ref_lit));
      std::string key = token.DebugString();
      full_stats_map[key] = stats;
      print_summary(
          absl::StrCat("Full execution (target=", key, ")"), stats);
    }
    const BenchmarkStats* single_full_stats =
        policy.IsSingleBackend()
            ? &full_stats_map[policy.unique_tokens().front().DebugString()]
            : nullptr;

    struct FragmentBenchmarkResult {
      int chunk_size;
      BenchmarkStats stats;
    };
    std::vector<FragmentBenchmarkResult> fragment_results;

    for (int chunk_size : chunk_sizes) {
      TF_ASSIGN_OR_RETURN(auto fragments,
                          SplitModuleWithChunkSize(*module, chunk_size));
      if (fragments.empty()) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(auto compiled_frags,
                          CompileFragments(fragments, "sapphirerapids",
                                           policy, chunk_size, cpu_client,
                                           gpu_client));

      bool has_gpu = ContainsGpuFragments(compiled_frags);

      std::shared_ptr<xla::Literal> fragmented_out;
      if (!has_gpu) {
        TF_RETURN_IF_ERROR(RunFragmentsSequentiallyCpuOnly(
            *module, compiled_frags, absl::MakeSpan(cpu_entry_ptrs),
            &fragmented_out));
      } else {
        TF_ASSIGN_OR_RETURN(auto verification_params,
                            InitializeParameterStates(input_span, cpu_device));
        TF_RETURN_IF_ERROR(RunFragmentsSequentiallyMultiBackend(
            *module, compiled_frags, &verification_params, &fragmented_out));
      }
      auto frag_compare =
          literal_comparison::Near(ref_lit, *fragmented_out,
                                   ErrorSpec(1e-1, 1e-1), std::nullopt, nullptr);
      if (!frag_compare.ok()) {
        std::cout << "Fragmented output does not match reference for chunk size "
                  << chunk_size << std::endl;
        LogLiteralMismatch(
            absl::StrCat("Fragmented (chunk=", chunk_size, ", policy=",
                         DescribePolicy(policy), ")"),
            ref_lit, *fragmented_out);
      } else {
        std::cout << "Fragmented output matches reference for chunk size "
                  << chunk_size << std::endl;
      }

      std::vector<ParameterState> reusable_params;
      if (has_gpu) {
        TF_ASSIGN_OR_RETURN(auto params,
                            InitializeParameterStates(input_span, cpu_device));
        reusable_params = std::move(params);
      }

      auto run_fragmented_once = [&]() -> absl::Status {
        if (!has_gpu) {
          return RunFragmentsSequentiallyCpuOnly(
              *module, compiled_frags, absl::MakeSpan(cpu_entry_ptrs),
              /*literal_out=*/nullptr);
        }
        return RunFragmentsSequentiallyMultiBackend(
            *module, compiled_frags, &reusable_params,
            /*literal_out=*/nullptr);
      };

      TF_ASSIGN_OR_RETURN(
          BenchmarkStats stats,
          BenchmarkExecution(
              absl::StrCat("Fragmented (policy=", DescribePolicy(policy),
                           ", chunk=", chunk_size, ")"),
              run_fragmented_once,
              /*skip_first_sample=*/has_gpu));
      fragment_results.push_back({chunk_size, stats});
    }

    for (const auto& frag_result : fragment_results) {
      print_summary(absl::StrCat("Fragmented chunk=", frag_result.chunk_size),
                    frag_result.stats);
      if (single_full_stats && frag_result.stats.mean_ms > 0.0) {
        std::cout << "  full/fragment mean ratio (chunk "
                  << frag_result.chunk_size << "): "
                  << single_full_stats->mean_ms / frag_result.stats.mean_ms
                  << "x" << std::endl;
      } else if (!policy.IsSingleBackend()) {
        std::cout << "  full/fragment mean ratio (chunk "
                  << frag_result.chunk_size
                  << "): N/A (multi-backend policy)" << std::endl;
      }
    }
  }

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: " << argv[0]
              << " <hlo_file> <backend_policy> <io_prefix> [--enable_gpu|--disable_gpu]"
              << std::endl;
    std::cerr << "  <hlo_file>: Path to the HLO text file." << std::endl;
    std::cerr << "  <backend_policy>: CPU feature experiments (e.g. 'all' or "
                 "'cpu:all|gpu')."
              << std::endl;
    std::cerr << "  <io_prefix>: Prefix path for input/output literal files."
              << std::endl;
    std::cerr << "  Optional fourth arg toggles GPU splitting (default: off)."
              << std::endl;
    return 1;
  }
  std::string hlo_file = argv[1];
  std::string features = argv[2];
  std::string io_prefix = argv[3];
  bool enable_gpu_backend = false;
  if (argc == 5) {
    std::string flag = argv[4];
    std::string lowered = absl::AsciiStrToLower(flag);
    if (lowered == "--enable_gpu" || lowered == "enable_gpu" ||
        lowered == "1" || lowered == "true") {
      enable_gpu_backend = true;
    } else if (lowered == "--disable_gpu" || lowered == "disable_gpu" ||
               lowered == "0" || lowered == "false") {
      enable_gpu_backend = false;
    } else {
      std::cerr << "Unrecognized GPU toggle '" << flag
                << "'. Use --enable_gpu/--disable_gpu or true/false."
                << std::endl;
      return 1;
    }
  }

  absl::Status status =
      RunAotCompilationExample(hlo_file, features, io_prefix,
                               enable_gpu_backend);
  if (!status.ok()) {
    std::cerr << "Error: " << status.ToString() << std::endl;
    return 1;
  }
  return 0;
}
