#ifndef XLA_HLO_TOOLS_HLO_AOT_COMMON_H_
#define XLA_HLO_TOOLS_HLO_AOT_COMMON_H_

#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"

struct BenchmarkStats {
  double mean_ms = 0.0;
  double stddev_ms = 0.0;
  double ci_half_width_ms = 0.0;
  int runs = 0;
};

BenchmarkStats ComputeBenchmarkStats(const std::vector<double>& samples);

void PrintBenchmarkSummary(absl::string_view label,
                           const BenchmarkStats& stats);

tsl::StatusOr<xla::PjRtDevice*> GetDefaultDevice(xla::PjRtClient* client);

tsl::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>>
PrepareArgumentBuffers(absl::Span<const xla::Literal> literals,
                       xla::PjRtDevice* device);

tsl::Status RunExecutableOnce(
    xla::PjRtLoadedExecutable* executable,
    absl::Span<xla::PjRtBuffer* const> entry_param_buffers,
    xla::PjRtDevice* device,
    const xla::ExecuteOptions& exec_opts,
    std::shared_ptr<xla::Literal>* literal_out);

tsl::StatusOr<BenchmarkStats> BenchmarkExecuteSharded(
    xla::PjRtLoadedExecutable* executable,
    absl::Span<xla::PjRtBuffer* const> entry_param_buffers,
    xla::PjRtDevice* device,
    const xla::ExecuteOptions& exec_opts,
    int max_runs = 20);

#endif  // XLA_HLO_TOOLS_HLO_AOT_COMMON_H_
