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
#include "xla/hlo/ir/hlo_clone_context.h"
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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/status_macros.h"

using namespace xla;
using xla::HloModule;
using xla::HloComputation;
using xla::HloInstruction;

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
  for (double v : samples) sum += v;
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
        1.96 * stats.stddev_ms /
        std::sqrt(static_cast<double>(samples.size()));
  }
  return stats;
}

void PrintBenchmarkSummary(absl::string_view label,
                           const BenchmarkStats& stats) {
  std::cout << label << " mean: " << stats.mean_ms
            << " ms (stddev " << stats.stddev_ms << " ms, 95% CI +/-"
            << stats.ci_half_width_ms << " ms over " << stats.runs << " runs)"
            << std::endl;
}

tsl::StatusOr<xla::PjRtDevice*> GetDefaultDevice(xla::PjRtClient* client) {
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Requested device but client is null"));
  }
  const auto& devices = client->devices();
  if (devices.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("No devices available "));
  }
  return devices.front();
}

tsl::StatusOr<std::unique_ptr<HloModule>> LoadModuleFromFile(
    const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open HLO file: ", path));
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  TF_ASSIGN_OR_RETURN(
      auto module,
      ParseAndReturnUnverifiedModule(buffer.str(), HloModuleConfig()));
  return module;
}

tsl::StatusOr<std::vector<xla::Literal>> GenerateInputLiterals(
    const xla::HloModule& module) {
  return xla::MakeFakeArguments(&module, /*pseudo_random=*/true);
}

tsl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> UploadLiteralToDevice(
    const xla::Literal& literal, xla::PjRtDevice* device) {
  TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                      device->default_memory_space());
  TF_ASSIGN_OR_RETURN(auto buffer,
                      device->client()->BufferFromHostLiteral(literal,
                                                              memory_space));
  TF_RETURN_IF_ERROR(buffer->GetReadyFuture().Await());
  return buffer;
}

tsl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
PrepareArgumentBuffers(absl::Span<const xla::Literal> literals,
                       xla::PjRtDevice* device) {
  std::vector<std::unique_ptr<xla::PjRtBuffer>> buffers;
  buffers.reserve(literals.size());
  for (const auto& literal : literals) {
    TF_ASSIGN_OR_RETURN(auto buf, UploadLiteralToDevice(literal, device));
    buffers.push_back(std::move(buf));
  }
  return buffers;
}

tsl::StatusOr<std::shared_ptr<xla::Literal>> ExecuteModuleOnCpu(
    const xla::HloModule& module,
    absl::Span<const xla::Literal> input_literals) {
  xla::CompileOptions compile_options;
  xla::CpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> cpu_client_holder,
                      xla::GetXlaPjrtCpuClient(client_options));
  auto* cpu_client =
      tsl::down_cast<xla::PjRtCpuClient*>(cpu_client_holder.get());

  xla::XlaComputation computation(module.ToProto());
  auto aot_options = std::make_unique<xla::cpu::CpuAotCompilationOptions>(
      /*triple=*/"x86_64-unknown-linux-gnu",
      /*cpu_name=*/"sapphirerapids",
      /*features=*/"",
      /*entry_point_name=*/"main.1",
      xla::cpu::CpuAotCompilationOptions::RelocationModel::Static);

  TF_ASSIGN_OR_RETURN(auto executable,
                      cpu_client->CompileAheadOfTimeAndLoad(
                          computation, compile_options, *aot_options));
  TF_ASSIGN_OR_RETURN(xla::PjRtDevice * device,
                      GetDefaultDevice(cpu_client));

  TF_ASSIGN_OR_RETURN(auto arg_buffers,
                      PrepareArgumentBuffers(input_literals, device));
  std::vector<xla::PjRtBuffer*> arg_ptrs;
  arg_ptrs.reserve(arg_buffers.size());
  for (auto& buffer : arg_buffers) {
    arg_ptrs.push_back(buffer.get());
  }

  xla::ExecuteOptions exec_opts;
  exec_opts.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;
  exec_opts.untuple_result = true;
  exec_opts.arguments_are_tupled = false;

  TF_ASSIGN_OR_RETURN(auto outputs,
                      executable->ExecuteSharded(arg_ptrs, device, exec_opts));
  if (outputs.empty()) {
    return absl::InternalError("Executable returned no outputs");
  }
  TF_RETURN_IF_ERROR(outputs[0]->GetReadyFuture().Await());
  TF_ASSIGN_OR_RETURN(auto literal, outputs[0]->ToLiteralSync());
  return literal;
}

tsl::Status RunExecutableOnce(
    xla::PjRtLoadedExecutable* executable,
    absl::Span<xla::PjRtBuffer* const> entry_param_buffers,
    xla::PjRtDevice* device,
    const xla::ExecuteOptions& exec_opts,
    std::shared_ptr<xla::Literal>* literal_out) {

  TF_ASSIGN_OR_RETURN(
      auto result_buffers,
      executable->ExecuteSharded(entry_param_buffers, device, exec_opts));
  if (result_buffers.empty()) {
    return absl::InternalError("Execution returned no outputs");
  }
  TF_RETURN_IF_ERROR(result_buffers[0]->GetReadyFuture().Await());
  TF_ASSIGN_OR_RETURN(auto literal, result_buffers[0]->ToLiteralSync());
  *literal_out = literal;
  return absl::OkStatus();
}

tsl::StatusOr<std::shared_ptr<xla::Literal>> ExecuteModuleOnGpu(
    const xla::HloModule& module,
    absl::Span<const xla::Literal> input_literals) {
  xla::CompileOptions compile_options;
  xla::GpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> gpu_client_holder,
                      xla::GetXlaPjrtGpuClient(client_options));
  xla::PjRtClient* gpu_client = gpu_client_holder.get();


  xla::XlaComputation computation(module.ToProto());
  TF_ASSIGN_OR_RETURN(auto executable,
                      gpu_client->CompileAndLoad(computation, compile_options));
  TF_ASSIGN_OR_RETURN(xla::PjRtDevice * device,
                      GetDefaultDevice(gpu_client));

  TF_ASSIGN_OR_RETURN(auto arg_buffers,
                      PrepareArgumentBuffers(input_literals, device));
  std::vector<xla::PjRtBuffer*> arg_ptrs;
  arg_ptrs.reserve(arg_buffers.size());
  for (auto& buf : arg_buffers) {
    arg_ptrs.push_back(buf.get());
  }

  xla::ExecuteOptions exec_opts;
  exec_opts.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;
  exec_opts.untuple_result = true;
  exec_opts.arguments_are_tupled = false;

  TF_ASSIGN_OR_RETURN(auto outputs,
                      executable->ExecuteSharded(arg_ptrs, device, exec_opts));
  if (outputs.empty()) {
    return absl::InternalError("Executable returned no outputs");
  }
  TF_RETURN_IF_ERROR(outputs[0]->GetReadyFuture().Await());
  TF_ASSIGN_OR_RETURN(auto literal, outputs[0]->ToLiteralSync());
  return literal;
}

absl::Status RunOnce(const std::string& hlo_file) {
  TF_ASSIGN_OR_RETURN(auto module, LoadModuleFromFile(hlo_file));
  TF_ASSIGN_OR_RETURN(auto inputs, GenerateInputLiterals(*module));
  TF_ASSIGN_OR_RETURN(auto output,
                      ExecuteModuleOnCpu(*module, absl::MakeConstSpan(inputs)));
  std::cout << "Output shape: "
            << xla::ShapeUtil::HumanString(output->shape()) << std::endl;
  std::cout << output->ToStringWithoutShape() << std::endl;
  return absl::OkStatus();
}

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cerr << "Usage: " << argv[0]
              << " <hlo_file> "
              << std::endl;
    return 1;
  }
  std::string hlo_file = argv[1];


  absl::Status status = RunOnce(argv[1]);
  if(!status.ok()) {
    std::cerr << "Error: " << status.ToString() << std::endl;
    return 1;
  }


  return 0;
}