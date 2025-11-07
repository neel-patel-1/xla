// hlo_aot.cc
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <vector>
#include "absl/strings/string_view.h"
#include "absl/strings/str_split.h"
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
  std::unique_ptr<HloModule>  module;
  const xla::HloInstruction* original_instr;
  std::vector<const xla::HloInstruction*> original_operands;
};

// Small holder tying a fragment to its compiled executable.
struct CompiledFragment {
  InstructionFragment frag;
  std::unique_ptr<xla::PjRtLoadedExecutable> exec;
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

absl::StatusOr<std::vector<InstructionFragment>>
SplitModulePerInstruction(const xla::HloModule& module) {
  const xla::HloComputation* entry = module.entry_computation();

  // Get a topological order; MakeInstructionPostOrder is fine.
  std::vector<xla::HloInstruction*> order = entry->MakeInstructionPostOrder();

  std::vector<InstructionFragment> fragments;
  fragments.reserve(order.size());

  // Base config we can reuse for each cloned module.
  xla::HloModuleConfig base_cfg = module.config();

  for (xla::HloInstruction* instr : order) {
    std::string mod_name = absl::StrCat("frag_", instr->name());

    if (instr->opcode() == xla::HloOpcode::kParameter) {
      continue;
    }

    xla::HloModuleConfig cfg(base_cfg);
    cfg.clear_entry_computation_layout();
    auto frag = std::make_unique<xla::HloModule>(mod_name, cfg);

    xla::HloComputation::Builder b(absl::StrCat(mod_name, "_entry"));

    std::vector<xla::HloInstruction*> new_operands;
    new_operands.reserve(instr->operand_count());
    std::vector<const xla::HloInstruction*> orig_ops;
    orig_ops.reserve(instr->operand_count());

    int param_idx = 0;
    for (xla::HloInstruction* op : instr->operands()) {
      // One parameter per operand, with the same shape as the operand.
      auto* p = b.AddInstruction(xla::HloInstruction::CreateParameter(
          param_idx, op->shape(),
          absl::StrCat("p", param_idx, "_for_", instr->name())));
      new_operands.push_back(p);
      orig_ops.push_back(op);
      ++param_idx;
    }

    // Now clone this instruction using the new operands as inputs.
    // CloneWithNewOperands preserves opcode + attributes.
    xla::HloInstruction* cloned =
        b.AddInstruction(instr->CloneWithNewOperands(instr->shape(),
                                                     new_operands));

    // Build computation with 'cloned' as ROOT.
    xla::HloComputation* comp =
        frag->AddEntryComputation(b.Build(cloned));
    frag->mutable_config().SetDefaultComputationLayout(
        comp->ComputeProgramShape());
    (void)comp;  // silence unused warning if not used

    InstructionFragment info;
    info.module = std::move(frag);
    info.original_instr = instr;
    info.original_operands = std::move(orig_ops);

    fragments.push_back(std::move(info));
  }

  return fragments;
}

// Compile each fragment with maybe different feature sets.
// For simplicity, this uses the same features for all; you can vary it
// by index or opcode.
absl::StatusOr<std::vector<CompiledFragment>>
CompileFragments(const std::vector<InstructionFragment>& fragments,
                 const std::string& cpu_name,
                 const std::string& features,
                 xla::PjRtCpuClient* cpu_client) {

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
    cf.frag = InstructionFragment{
        /*module=*/std::unique_ptr<xla::HloModule>(
            frag.module->Clone()),  // or keep by pointer/ref
        /*original_instr=*/frag.original_instr,
        /*original_operands=*/frag.original_operands};
    cf.exec = std::move(exec);
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
    int param_idx = instr->parameter_number();
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

  // Execute fragments in the same order as SplitModulePerInstruction.
  for (const auto& cf : compiled_frags) {
    const InstructionFragment& frag = cf.frag;
    const xla::HloInstruction* orig_instr = frag.original_instr;

    // 1) Build the arg list for this fragment from value_map and params.
    std::vector<xla::PjRtBuffer*> arg_ptrs;
    arg_ptrs.reserve(frag.original_operands.size());

    for (const xla::HloInstruction* op : frag.original_operands) {
      if (op->opcode() == xla::HloOpcode::kParameter) {
        int param_idx = op->parameter_number();
        arg_ptrs.push_back(entry_param_buffers[param_idx].get());
      } else {
        // internal value, must have been produced by an earlier fragment
        auto it = value_map.find(op);
        if (it == value_map.end()) {
          return absl::InternalError(
              absl::StrCat("Missing value for operand ", op->name(),
                           " of instruction ", orig_instr->name()));
        }
        arg_ptrs.push_back(it->second.get());
      }
    }

    // 2) Execute the fragment
    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<xla::PjRtBuffer>> res,
        cf.exec->ExecuteSharded(arg_ptrs, device, exec_opts));
    TF_RETURN_IF_ERROR(res[0]->GetReadyFuture().Await());

    // 3) Store its output in value_map
    value_map[orig_instr] = std::move(res[0]);
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

absl::Status RunAotCompilationExample(std::string hlo_file, std::string features_str, std::string io_prefix) {
  xla::CompileOptions compile_options;
  llvm::StringMap<bool, llvm::MallocAllocator> host_machine_features = llvm::sys::getHostCPUFeatures();

  // Read HLO from file
  std::ifstream in_file(hlo_file);
  if (!in_file) {
    return absl::InvalidArgumentError(absl::StrCat("Failed to open HLO file: ", hlo_file));
  }
  std::stringstream buffer;
  buffer << in_file.rdbuf();
  std::string hlo = buffer.str();

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnUnverifiedModule(
                          hlo,
                          HloModuleConfig() /* unused */));

  TF_ASSIGN_OR_RETURN(std::vector<InstructionFragment> frags,
                      SplitModulePerInstruction(*module));

  xla::CpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(client_options));
  auto* cpu_client = tsl::down_cast<xla::PjRtCpuClient*>(client.get());

  TF_ASSIGN_OR_RETURN(std::vector<CompiledFragment> compiled_frags,
                       CompileFragments(frags, "skylake-avx512", features_str, cpu_client));

  int num_params = module->entry_computation_layout().parameter_count();
  std::vector<xla::Literal> input_lits;
  input_lits.reserve(num_params);

  for (int i = 0; i < num_params; ++i){
    std::string path =
        absl::StrCat(io_prefix, "/input_", i, ".litpb");
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

  std::shared_ptr<xla::Literal> fragmented_out;
  TF_RETURN_IF_ERROR(RunFragmentsSequentially(
      *module, compiled_frags, args_buffers, client.get(), &fragmented_out));
  EXPECT_TRUE(xla::LiteralTestUtil::NearOrEqual(ref_lit, *fragmented_out,
                                                ErrorSpec(1e-5, 1e-5)));
  std::cout << "Fragmented execution matches reference." << std::endl;

  auto run_fragmented_once = [&]() -> absl::Status {
    return RunFragmentsSequentially(*module, compiled_frags, args_buffers,
                                    client.get(), /*literal_out=*/nullptr);
  };

  TF_ASSIGN_OR_RETURN(
      BenchmarkStats fragmented_stats,
      BenchmarkExecution("Fragmented execution", run_fragmented_once));

  std::unique_ptr<xla::AotCompilationOptions> aot_options;

  std::vector<std::string> compile_machine_features = absl::StrSplit(features_str, ',');
  if (features_str == "all"){
    llvm::StringMap<bool> host_machine_features = llvm::sys::getHostCPUFeatures();
    compile_machine_features.clear();
    for (const auto& feature : host_machine_features) {
      if (feature.second) {
        compile_machine_features.push_back(feature.first().str());
      }
    }
  }

  aot_options = std::make_unique<xla::cpu::CpuAotCompilationOptions>(
      /*triple=*/"x86_64-unknown-linux-gnu", /*cpu_name=*/"skylake-avx512",
      /*features=*/absl::StrJoin(compile_machine_features, ","),
      /*entry_point_name=*/"main.1",
      /*relocation_model=*/xla::cpu::CpuAotCompilationOptions::RelocationModel::Static
  );

  xla::XlaComputation computation(module->ToProto());
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  TF_ASSIGN_OR_RETURN( executable,
    cpu_client->CompileAheadOfTimeAndLoad(
    computation,
    compile_options,
    *aot_options
  ));


  ExecuteOptions execute_options;
  execute_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;

  TF_ASSIGN_OR_RETURN(
      auto result_buffers,
      executable->ExecuteSharded(arg_ptrs, device, execute_options));
  TF_RETURN_IF_ERROR(result_buffers[0]->GetReadyFuture().Await());
  TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> full_out,
                      result_buffers[0]->ToLiteralSync());
  EXPECT_TRUE(
      xla::LiteralTestUtil::NearOrEqual(ref_lit, *full_out, ErrorSpec(1e-5, 1e-5)));
  std::cout << "Full executable output matches reference." << std::endl;

  auto run_full_once = [&]() -> absl::Status {
    TF_ASSIGN_OR_RETURN(
        auto iteration_buffers,
        executable->ExecuteSharded(arg_ptrs, device, execute_options));
    TF_RETURN_IF_ERROR(iteration_buffers[0]->GetReadyFuture().Await());
    return absl::OkStatus();
  };

  TF_ASSIGN_OR_RETURN(
      BenchmarkStats full_stats,
      BenchmarkExecution("Full execution", run_full_once));

  auto print_summary = [](absl::string_view label,
                          const BenchmarkStats& stats) {
    std::cout << label << " mean: " << stats.mean_ms
              << " ms (stddev " << stats.stddev_ms << " ms, 95% CI +/-"
              << stats.ci_half_width_ms << " ms over " << stats.runs
              << " runs)" << std::endl;
  };

  print_summary("Fragmented execution", fragmented_stats);
  print_summary("Full execution", full_stats);
  if (fragmented_stats.mean_ms > 0.0) {
    std::cout << "Full / fragmented mean time ratio: "
              << full_stats.mean_ms / fragmented_stats.mean_ms << "x"
              << std::endl;
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
