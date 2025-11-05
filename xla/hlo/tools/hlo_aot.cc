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
#include <iostream>

int main(int argc, char** argv) {

  absl::string_view hlo = R"(
    HloModule dot_f32

    ENTRY e {
      a = f32[2,2] parameter(0)
      b = f32[2,2] parameter(1)
      ROOT c = f32[2,2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";


  llvm::StringMap<bool, llvm::MallocAllocator> host_machine_features = llvm::sys::getHostCPUFeatures();
  auto compile_machine_features = absl::StrSplit("avx512f,avx512vl", ',');

  std::unique_ptr<xla::AotCompilationOptions> aot_options;
  aot_options = std::make_unique<xla::cpu::CpuAotCompilationOptions>(
      /*triple=*/"x86_64-unknown-linux-gnu", /*cpu_name=*/"skylake-avx512",
      /*features=*/absl::StrJoin(compile_machine_features, ","),
      /*entry_point_name=*/"dot_f32",
      /*relocation_model=*/xla::cpu::CpuAotCompilationOptions::RelocationModel::Static);




  return 0;
}
