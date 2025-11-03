// hlo_aot.cc
#include <fstream>
#include "absl/strings/string_view.h"
#include "llvm/Support/CodeGen.h"  // For Reloc
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compile_only_service.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/platform_manager.h"  // Added for PlatformManager

int main(int argc, char** argv) {
  const std::string in_hlo = argv[1];
  const std::string out_o  = argv[2];
  const std::string triple = argv[3];     // e.g. "x86_64-pc-linux-gnu"
  const std::string cpu    = argv[4];     // e.g. "skylake-avx512"
  const std::string features = argv[5];   // e.g. "+avx512f,+avx512dq"

  using RM = xla::cpu::CpuAotCompilationOptions::RelocationModel;

  // 1) Parse HLO
  auto mod = xla::ParseAndReturnUnverifiedModule(in_hlo).value(); // or read file then parse

  // 2) Describe argument/result layouts for AOT
  xla::CompileOnlyService::AotXlaComputationInstance instance;
  instance.computation = mod->ToProto();
  std::vector<const xla::Shape*> arg_layouts;
  for (int i = 0; i < mod->entry_computation_layout().parameter_count(); ++i) {
    arg_layouts.push_back(&mod->entry_computation_layout().parameter_shape(i));
  }
  instance.argument_layouts = std::move(arg_layouts);
  instance.result_layout = mod->result_shape();

  // 3) Build AOT options for CPU (triple/CPU/features)
  xla::cpu::CpuAotCompilationOptions cpu_opts(triple, cpu, features, "entry", RM::Static);
  xla::AotCompilationOptions& aot_opts = cpu_opts;

  // 4) Create a compile-only service for CPU and AOT-compile
  auto platform = stream_executor::PlatformManager::PlatformWithName("CPU").value();
  auto service = xla::CompileOnlyService::NewService(platform).value();

  std::unique_ptr<xla::AotCompilationMetadata> metadata;
  auto results = service->CompileAheadOfTime({instance}, aot_opts, &metadata).value();

  // 5) Extract object bytes and write .o
  auto* cpu_res =
      static_cast<xla::cpu::CpuAotCompilationResult*>(results[0].get());
  absl::string_view *obj = cpu_res->obj_files().data();
  std::ofstream(out_o, std::ios::binary).write(obj->data(), obj->size());
  // std::ofstream(out_o, std::ios::binary).write(obj.data(), obj.size());
}
