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

int main(int argc, char** argv) {
  if (argc < 1) {
    std::cerr << "Usage: " << argv[0]
              << " <hlo_file> "
              << std::endl;
    return 1;
  }

  std::string hlo_file = argv[1];

  return 0;
}