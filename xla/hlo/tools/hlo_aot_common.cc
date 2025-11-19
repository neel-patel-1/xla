#include "xla/hlo/tools/hlo_aot_common.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/status_macros.h"

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

namespace {

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

}  // namespace

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
  if (literal_out != nullptr) {
    *literal_out = literal;
  }
  return absl::OkStatus();
}

tsl::StatusOr<BenchmarkStats> BenchmarkExecuteSharded(
    xla::PjRtLoadedExecutable* executable,
    absl::Span<xla::PjRtBuffer* const> entry_param_buffers,
    xla::PjRtDevice* device,
    const xla::ExecuteOptions& exec_opts,
    int max_runs) {
  constexpr double kMaxRelativeCi = 0.05;
  std::vector<double> samples_ms;
  for (int run = 0; run < max_runs; ++run) {
    absl::Time start = absl::Now();
    TF_RETURN_IF_ERROR(
        RunExecutableOnce(executable, entry_param_buffers, device, exec_opts,
                          /*literal_out=*/nullptr));
    absl::Time end = absl::Now();
    double duration_ms = absl::ToDoubleMilliseconds(end - start);
    samples_ms.push_back(duration_ms);
    if (samples_ms.size() < 2) {
      continue;
    }
    BenchmarkStats stats = ComputeBenchmarkStats(samples_ms);
    double relative_ci =
        stats.ci_half_width_ms / std::max(stats.mean_ms, 1e-8);
    if (relative_ci <= kMaxRelativeCi) {
      return stats;
    }
  }
  return ComputeBenchmarkStats(samples_ms);
}
