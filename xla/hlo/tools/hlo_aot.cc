// hlo_aot.cc
#include <fstream>
#include <sstream>
#include "absl/strings/string_view.h"
#include "absl/strings/str_split.h"
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
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/platform/statusor.h"  // Add for TF_ASSIGN_OR_RETURN
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tests/test_utils.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"
#include "xla/error_spec.h"

#include <iostream>

using namespace xla;

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

  xla::CpuClientOptions client_options;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                      xla::GetXlaPjrtCpuClient(client_options));

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
  auto* cpu_client = tsl::down_cast<xla::PjRtCpuClient*>(client.get());
  std::unique_ptr<xla::PjRtLoadedExecutable> executable;
  TF_ASSIGN_OR_RETURN( executable,
    cpu_client->CompileAheadOfTimeAndLoad(
    computation,
    compile_options,
    *aot_options
  ));


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

  ExecuteOptions execute_options;
  execute_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;

  TF_ASSIGN_OR_RETURN(
    auto result_buffers,
    executable->ExecuteSharded(arg_ptrs, device, execute_options));

  TF_RETURN_IF_ERROR(result_buffers[0]->GetReadyFuture().Await());

  TF_ASSIGN_OR_RETURN(
    std::shared_ptr<xla::Literal> out_lit,
    result_buffers[0]->ToLiteralSync());

  // Load reference from JAX
  std::string ref_path =
      absl::StrCat(io_prefix, "/output_0.ref.litpb");
  TF_ASSIGN_OR_RETURN(xla::Literal ref_lit,
                      LoadLiteralFromProtoFile(ref_path));

  EXPECT_TRUE(xla::LiteralTestUtil::NearOrEqual(ref_lit, *out_lit, ErrorSpec(1e-5, 1e-5)));


  std::cout << "Output matches reference." << std::endl;

  std::vector<std::unique_ptr<PjRtBuffer>> results;

  auto run_benchmark_once = [&]() -> absl::Status {
    results =
        executable->ExecuteSharded(arg_ptrs, device, execute_options)
            .value();
    CHECK_OK(results[0]->GetReadyFuture().Await());
    return absl::OkStatus();
  };

  // Run benchmark multiple times to achieve 95% CI within 5% of mean
  const int max_runs = 1000;  // Safety limit
  std::vector<double> times;
  times.reserve(max_runs);

  for (int i = 0; i < max_runs; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    TF_RETURN_IF_ERROR(run_benchmark_once());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    times.push_back(diff.count());
    std::cout<< "Run " << i+1 << ": " << diff.count() * 1000 << " ms" <<std::endl;

    // Compute mean and std dev
    double sum = 0.0;
    for (double t : times) sum += t;
    double mean = sum / times.size();
    double var = 0.0;
    for (double t : times) var += (t - mean) * (t - mean);
    double std_dev = std::sqrt(var / (times.size() - 1));
    double ci_half_width = 1.96 * std_dev / std::sqrt(times.size());
    double relative_ci = ci_half_width / mean;

    if (relative_ci <= 0.05) {  // 5% of mean
      std::cout << "Achieved 95% CI within 5% of mean after " << times.size() << " runs." << std::endl;
      std::cout << "Mean time: " << mean * 1000 << " ms, Std dev: " << std_dev * 1000 << " ms, CI half-width: " << ci_half_width * 1000 << " ms" << std::endl;
      break;
    }
  }

  if (times.size() == max_runs) {
    std::cout << "Reached max runs without achieving CI target." << std::endl;
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
