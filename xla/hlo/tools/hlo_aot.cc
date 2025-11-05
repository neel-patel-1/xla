// hlo_aot.cc
#include <fstream>
#include <sstream>
#include "absl/strings/string_view.h"
#include "llvm/Support/CodeGen.h"  // For Reloc
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compile_only_service.h"
#include "xla/service/cpu/cpu_aot_compilation_result.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/platform_manager.h"  // Added for PlatformManager

using namespace xla;
int main(int argc, char** argv) {

  absl::string_view hlo = R"(
    HloModule dot_f32

    ENTRY e {
      a = f32[2,2] parameter(0)
      b = f32[2,2] parameter(1)
      ROOT c = f32[2,2] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";
  std::unique_ptr<AotCompilationOptions> aot_options;




  return 0;
}
