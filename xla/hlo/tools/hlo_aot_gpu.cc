#include "xla/hlo/tools/hlo_aot_gpu.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/tools/hlo_aot_common.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "xla/status_macros.h"

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

  TF_ASSIGN_OR_RETURN(
      BenchmarkStats stats,
      BenchmarkExecuteSharded(executable.get(), absl::MakeSpan(arg_ptrs),
                              device, exec_opts));
  PrintBenchmarkSummary("[GPU] ExecuteSharded", stats);

  TF_ASSIGN_OR_RETURN(auto outputs,
                      executable->ExecuteSharded(arg_ptrs, device, exec_opts));
  if (outputs.empty()) {
    return absl::InternalError("Executable returned no outputs");
  }
  TF_RETURN_IF_ERROR(outputs[0]->GetReadyFuture().Await());
  TF_ASSIGN_OR_RETURN(auto literal, outputs[0]->ToLiteralSync());
  return literal;
}
