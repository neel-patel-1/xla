// hlo_aot.cc
#include <fstream>
#include <sstream>
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
#include "xla/shape_util.h"
#include <iostream>

absl::Status RunAotCompilationExample() {
  xla::CompileOptions compile_options;
  llvm::StringMap<bool, llvm::MallocAllocator> host_machine_features = llvm::sys::getHostCPUFeatures();
  absl::string_view hlo = R"(
    HloModule dot_f32

    ENTRY e {
      a = f32[2,2] parameter(0)
      b = f32[2,2] parameter(1)
      ROOT c = f32[2,2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  xla::CpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(client_options));

  xla::PjRtDevice* device = client->devices().front();
  TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                      device->default_memory_space());
  std::unique_ptr<xla::AotCompilationOptions> aot_options;

  std::unique_ptr<xla::HloModule> module = std::make_unique<xla::VerifiedHloModule>(
      "test", xla::HloModuleConfig() /* unused */,
      /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      xla::ShapeUtil::ByteSizeOfElements);

  auto compile_machine_features = absl::StrSplit("avx512f,avx512vl", ',');
  aot_options = std::make_unique<xla::cpu::CpuAotCompilationOptions>(
      /*triple=*/"x86_64-unknown-linux-gnu", /*cpu_name=*/"skylake-avx512",
      /*features=*/absl::StrJoin(compile_machine_features, ","),
      /*entry_point_name=*/"dot_f32",
      /*relocation_model=*/xla::cpu::CpuAotCompilationOptions::RelocationModel::Static
  );

  xla::XlaComputation computation(module->ToProto());
  auto* cpu_client = tsl::down_cast<xla::PjRtCpuClient*>(client.get());
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  TF_ASSIGN_OR_RETURN( executable,
    cpu_client->CompileAheadOfTimeAndLoad(
    computation,
    compile_options,
    *aot_options
  ));
}

int main(int argc, char** argv) {

  RunAotCompilationExample();
  return 0;
}
