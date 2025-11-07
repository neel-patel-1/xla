// hlo_aot.cc
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "llvm/Support/CodeGen.h"  // For Reloc
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compile_only_service.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/platform_manager.h"  // Added for PlatformManager
#include "llvm/TargetParser/Host.h"  // For
#include "llvm/ADT/StringMap.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
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

using namespace xla;
using xla::HloModule;
using xla::HloComputation;
using xla::HloInstruction;

struct InstructionFragment {
  std::unique_ptr<HloModule> module;
  std::vector<const xla::HloInstruction*> produced_instructions;
  std::vector<const xla::HloInstruction*> external_operands;
  std::string description;
};

// Small holder tying a fragment to its compiled executable.
struct CompiledFragment {
  InstructionFragment frag;
  std::unique_ptr<xla::PjRtLoadedExecutable> exec;
  std::string feature_set;
  int chunk_size = 0;
};

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
                 const std::string& cpu_name, const std::string& features,
                 int chunk_size, xla::PjRtCpuClient* cpu_client) {

  xla::CompileOptions compile_options;

  std::vector<CompiledFragment> compiled;
  compiled.reserve(fragments.size());

  for (const auto& frag : fragments) {
    xla::XlaComputation computation(frag.module->ToProto());

    auto entry_point = std::string(frag.module->entry_computation()->name());
    auto aot_opts = xla::cpu::CpuAotCompilationOptions(
        /*triple=*/"x86_64-unknown-linux-gnu",
        /*cpu_name=*/cpu_name,
        /*features=*/features,
        /*entry_point_name=*/entry_point,
        /*relocation_model=*/
        xla::cpu::CpuAotCompilationOptions::RelocationModel::Static);

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtLoadedExecutable> exec,
        cpu_client->CompileAheadOfTimeAndLoad(
            computation, compile_options, aot_opts));

    CompiledFragment cf;
    InstructionFragment stored;
    stored.module =
        std::unique_ptr<xla::HloModule>(frag.module->Clone());
    stored.produced_instructions = frag.produced_instructions;
    stored.external_operands = frag.external_operands;
    stored.description = frag.description;
    cf.frag = std::move(stored);
    cf.exec = std::move(exec);
    cf.feature_set = features;
    cf.chunk_size = chunk_size;
    compiled.push_back(std::move(cf));
  }

  return compiled;
}

absl::Status RunFragmentsSequentially(
    const xla::HloModule& original,
    const std::vector<CompiledFragment>& compiled_frags,
    const std::vector<std::unique_ptr<xla::PjRtBuffer>>& entry_param_buffers,
    xla::PjRtClient* client,
    std::shared_ptr<xla::Literal>* literal_out) {

  xla::PjRtDevice* device = client->devices().front();

  // Map original HloInstruction* -> computed device buffer.
  absl::flat_hash_map<const xla::HloInstruction*,
                      std::unique_ptr<xla::PjRtBuffer>>
      value_map;

  // Seed map with original parameters: param(i) -> entry_param_buffers[i]
  const xla::HloComputation* entry = original.entry_computation();
  for (xla::HloInstruction* instr : entry->parameter_instructions()) {
    // Clone the buffer pointer (just move a unique_ptr if you own them).
    // Here we assume entry_param_buffers already live long enough and
    // we just *point* to them:
    // (If you want ownership, wrap them differently.)
    // We'll store raw pointers for simplicity:
    // value_map[instr] = <PjRtBuffer*>;
    // In a real implementation, choose a consistent ownership model.
  }

  xla::ExecuteOptions exec_opts;
  exec_opts.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;

  // Execute fragments in the same order as they were generated.
  for (const auto& cf : compiled_frags) {
    const InstructionFragment& frag = cf.frag;

    // 1) Build the arg list for this fragment from value_map and params.
    std::vector<xla::PjRtBuffer*> arg_ptrs;
    arg_ptrs.reserve(frag.external_operands.size());

    for (const xla::HloInstruction* op : frag.external_operands) {
      if (op->opcode() == xla::HloOpcode::kParameter) {
        int param_idx = op->parameter_number();
        arg_ptrs.push_back(entry_param_buffers[param_idx].get());
      } else {
        // internal value, must have been produced by an earlier fragment
        auto it = value_map.find(op);
        if (it == value_map.end()) {
          return absl::InternalError(
              absl::StrCat("Missing value for operand ", op->name(),
                           " while executing fragment ", frag.description));
        }
        arg_ptrs.push_back(it->second.get());
      }
    }

    // 2) Execute the fragment
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<xla::PjRtBuffer>> exec_outputs,
        cf.exec->ExecuteSharded(arg_ptrs, device, exec_opts));
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
    absl::string_view label,
    const std::function<absl::Status()>& run_once) {
  constexpr int kMaxRuns = 1000;
  std::vector<double> samples;
  samples.reserve(kMaxRuns);

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

std::vector<std::string> ParseFeatureSets(const std::string& raw) {
  std::vector<std::string> sets;
  if (raw.empty()) {
    sets.push_back("");
    return sets;
  }
  for (absl::string_view part : absl::StrSplit(raw, ';')) {
    std::string cleaned = std::string(absl::StripAsciiWhitespace(part));
    if (!cleaned.empty()) {
      sets.push_back(cleaned);
    }
  }
  if (sets.empty()) {
    sets.push_back("");
  }
  return sets;
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

absl::Status RunAotCompilationExample(std::string hlo_file,
                                      std::string features_str,
                                      std::string io_prefix) {
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
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(client_options));
  auto* cpu_client = tsl::down_cast<xla::PjRtCpuClient*>(client.get());

  int num_params = module->entry_computation_layout().parameter_count();
  std::vector<xla::Literal> input_lits;
  input_lits.reserve(num_params);
  for (int i = 0; i < num_params; ++i) {
    std::string path = absl::StrCat(io_prefix, "/input_", i, ".litpb");
    TF_ASSIGN_OR_RETURN(auto lit, LoadLiteralFromProtoFile(path));
    input_lits.push_back(std::move(lit));
  }

  std::vector<std::unique_ptr<PjRtBuffer>> args_buffers;
  args_buffers.reserve(input_lits.size());
  xla::PjRtDevice* device = client->devices().front();
  TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                      device->default_memory_space());
  for (const xla::Literal& arg_lit : input_lits) {
    TF_ASSIGN_OR_RETURN(args_buffers.emplace_back(),
                        client->BufferFromHostLiteral(arg_lit, memory_space));
    TF_RETURN_IF_ERROR(args_buffers.back()->GetReadyFuture().Await());
  }

  std::vector<PjRtBuffer*> arg_ptrs;
  arg_ptrs.reserve(args_buffers.size());
  for (auto& buf : args_buffers) {
    arg_ptrs.push_back(buf.get());
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

  auto feature_sets = ParseFeatureSets(features_str);
  auto print_summary = [](absl::string_view label,
                          const BenchmarkStats& stats) {
    std::cout << label << " mean: " << stats.mean_ms
              << " ms (stddev " << stats.stddev_ms << " ms, 95% CI +/-"
              << stats.ci_half_width_ms << " ms over " << stats.runs << " runs)"
              << std::endl;
  };

  for (const std::string& feature_token : feature_sets) {
    std::string materialized_features =
        MaterializeFeatureString(feature_token, host_machine_features);
    std::cout << "=== Feature set: "
              << (materialized_features.empty() ? std::string("<default>")
                                                : materialized_features)
              << " ===" << std::endl;

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
                          CompileFragments(fragments, "skylake-avx512",
                                           materialized_features, chunk_size,
                                           cpu_client));

      std::shared_ptr<xla::Literal> fragmented_out;
      TF_RETURN_IF_ERROR(RunFragmentsSequentially(
          *module, compiled_frags, args_buffers, client.get(),
          &fragmented_out));
      EXPECT_TRUE(xla::LiteralTestUtil::NearOrEqual(ref_lit, *fragmented_out,
                                                    ErrorSpec(1e-5, 1e-5)));

      auto run_fragmented_once = [&]() -> absl::Status {
        return RunFragmentsSequentially(*module, compiled_frags, args_buffers,
                                        client.get(), /*literal_out=*/nullptr);
      };

      TF_ASSIGN_OR_RETURN(
          BenchmarkStats stats,
          BenchmarkExecution(
              absl::StrCat("Fragmented (features=", materialized_features,
                           ", chunk=", chunk_size, ")"),
              run_fragmented_once));
      fragment_results.push_back({chunk_size, stats});
    }

    auto aot_options = std::make_unique<xla::cpu::CpuAotCompilationOptions>(
        /*triple=*/"x86_64-unknown-linux-gnu", /*cpu_name=*/"skylake-avx512",
        /*features=*/materialized_features,
        /*entry_point_name=*/"main.1",
        /*relocation_model=*/
        xla::cpu::CpuAotCompilationOptions::RelocationModel::Static);

    xla::XlaComputation computation(module->ToProto());
    std::unique_ptr<xla::PjRtLoadedExecutable> executable;
    TF_ASSIGN_OR_RETURN(executable,
                        cpu_client->CompileAheadOfTimeAndLoad(
                            computation, compile_options, *aot_options));

    ExecuteOptions execute_options;
    execute_options.execution_mode =
        ExecuteOptions::ExecutionMode::kSynchronous;

    TF_ASSIGN_OR_RETURN(auto result_buffers,
                        executable->ExecuteSharded(arg_ptrs, device,
                                                   execute_options));
    TF_RETURN_IF_ERROR(result_buffers[0]->GetReadyFuture().Await());
    TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> full_out,
                        result_buffers[0]->ToLiteralSync());
    EXPECT_TRUE(xla::LiteralTestUtil::NearOrEqual(ref_lit, *full_out,
                                                  ErrorSpec(1e-5, 1e-5)));

    auto run_full_once = [&]() -> absl::Status {
      TF_ASSIGN_OR_RETURN(auto iteration_buffers,
                          executable->ExecuteSharded(arg_ptrs, device,
                                                     execute_options));
      TF_RETURN_IF_ERROR(iteration_buffers[0]->GetReadyFuture().Await());
      return absl::OkStatus();
    };

    TF_ASSIGN_OR_RETURN(BenchmarkStats full_stats,
                        BenchmarkExecution(
                            absl::StrCat("Full (features=", materialized_features,
                                         ")"),
                            run_full_once));

    print_summary("Full execution", full_stats);
    for (const auto& frag_result : fragment_results) {
      print_summary(absl::StrCat("Fragmented chunk=", frag_result.chunk_size),
                    frag_result.stats);
      if (frag_result.stats.mean_ms > 0.0) {
        std::cout << "  full/fragment mean ratio (chunk "
                  << frag_result.chunk_size << "): "
                  << full_stats.mean_ms / frag_result.stats.mean_ms << "x"
                  << std::endl;
      }
    }
  }

  return absl::OkStatus();
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <hlo_file> <cpu_features> <io_prefix>" << std::endl;
    std::cerr << "  <hlo_file>: Path to the HLO text file." << std::endl;
    std::cerr << "  <cpu_features>: Comma-separated CPU features to enable (or 'all' for all features)." << std::endl;
    std::cerr << "  <io_prefix>: Prefix path for input/output literal files." << std::endl;
    return 1;
  }
  std::string hlo_file = argv[1];
  std::string features = argv[2];
  std::string io_prefix = argv[3];

  absl::Status status = RunAotCompilationExample(hlo_file, features, io_prefix);
  if (!status.ok()) {
    std::cerr << "Error: " << status.ToString() << std::endl;
    return 1;
  }
  return 0;
}
