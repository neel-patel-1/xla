#include "xla/hlo/tools/hlo_aot_cpu.h"

#include <memory>
#include <utility>
#include <vector>

#include "tsl/platform/casts.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/compiler.h"
#include "xla/status_macros.h"

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

  TF_ASSIGN_OR_RETURN(
      BenchmarkStats stats,
      BenchmarkExecuteSharded(executable.get(), absl::MakeSpan(arg_ptrs),
                              device, exec_opts));
  PrintBenchmarkSummary("[CPU] ExecuteSharded", stats);

  TF_ASSIGN_OR_RETURN(auto outputs,
                      executable->ExecuteSharded(arg_ptrs, device, exec_opts));
  if (outputs.empty()) {
    return absl::InternalError("Executable returned no outputs");
  }
  TF_RETURN_IF_ERROR(outputs[0]->GetReadyFuture().Await());
  TF_ASSIGN_OR_RETURN(auto literal, outputs[0]->ToLiteralSync());
  return literal;
}
