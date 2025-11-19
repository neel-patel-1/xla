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
#include "xla/hlo/tools/hlo_aot_cpu.h"
#include "xla/hlo/tools/hlo_aot_gpu.h"

using namespace xla;
using xla::HloModule;
using xla::HloComputation;
using xla::HloInstruction;

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

tsl::StatusOr<std::shared_ptr<xla::Literal>> ExecuteModuleWithBackendAnnotations(
    const xla::HloModule& module, absl::Span<const xla::Literal> input_literals){
  xla::CompileOptions cpu_compile_options;
  xla::CpuClientOptions cpu_client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> cpu_client_holder,
                      xla::GetXlaPjrtCpuClient(cpu_client_options));
  auto* cpu_client =
      tsl::down_cast<xla::PjRtCpuClient*>(cpu_client_holder.get());

  xla::CompileOptions gpu_compile_options;
  xla::GpuClientOptions gpu_client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> gpu_client_holder,
                      xla::GetXlaPjrtGpuClient(gpu_client_options));
  xla::PjRtClient* gpu_client = gpu_client_holder.get();

}

absl::Status Run(const std::string& hlo_file ) {
  TF_ASSIGN_OR_RETURN(auto module, LoadModuleFromFile(hlo_file));
  TF_ASSIGN_OR_RETURN(auto inputs, GenerateInputLiterals(*module));
  TF_ASSIGN_OR_RETURN(
      auto output,
      ExecuteModuleOnCpu(*module, absl::MakeSpan(inputs)));
  std::cout << "CPU\n Output shape: "
            << xla::ShapeUtil::HumanString(output->shape()) << std::endl;

  TF_ASSIGN_OR_RETURN(
      auto gpu_output,
      ExecuteModuleOnGpu(*module, absl::MakeSpan(inputs)));
  std::cout << "GPU\n Output shape: "
            << xla::ShapeUtil::HumanString(gpu_output->shape()) << std::endl;
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


  absl::Status status = Run(argv[1]);
  if(!status.ok()) {
    std::cerr << "Error: " << status.ToString() << std::endl;
    return 1;
  }


  return 0;
}
