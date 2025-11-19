#ifndef XLA_HLO_TOOLS_HLO_AOT_GPU_H_
#define XLA_HLO_TOOLS_HLO_AOT_GPU_H_

#include <memory>

#include "absl/types/span.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"

tsl::StatusOr<std::shared_ptr<xla::Literal>> ExecuteModuleOnGpu(
    const xla::HloModule& module,
    absl::Span<const xla::Literal> input_literals);

#endif  // XLA_HLO_TOOLS_HLO_AOT_GPU_H_
