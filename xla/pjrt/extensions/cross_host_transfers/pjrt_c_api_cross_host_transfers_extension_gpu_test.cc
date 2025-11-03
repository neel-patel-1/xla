/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/ffi/api/ffi.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_test.h"
#include "xla/pjrt/c/pjrt_c_api_test_base.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/extensions/cross_host_transfers/pjrt_c_api_cross_host_transfers_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"

namespace pjrt {
namespace {

using ::testing::ElementsAreArray;

#ifdef TENSORFLOW_USE_ROCM
const bool kUnused = (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); },
                                                  /*platform_name=*/"rocm"),
                      true);
#else   // TENSORFLOW_USE_ROCM
const bool kUnused = (RegisterPjRtCApiTestFactory([]() { return GetPjrtApi(); },
                                                  /*platform_name=*/"cuda"),
                      true);
#endif  // TENSORFLOW_USE_ROCM

class PjrtCApiGpuCrossHostTransfersTest : public PjrtCApiTestBase {
 public:
  PjrtCApiGpuCrossHostTransfersTest()
      : PjrtCApiTestBase(GetPjrtApi(),
                         {{"visible_devices",
                           xla::PjRtValueType(std::vector<int64_t>{0, 1})}}) {}
};

TEST_F(PjrtCApiGpuCrossHostTransfersTest, SuccessfulTransfer) {
  auto api = GetPjrtApi();
  PJRT_CrossHostTransfers_Extension* cross_host_transfers_extension =
      pjrt::FindExtension<PJRT_CrossHostTransfers_Extension>(
          api, PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  ASSERT_NE(cross_host_transfers_extension, nullptr);
  ASSERT_NE(cross_host_transfers_extension
                ->PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice,
            nullptr);

  // Create a receive buffer.
  PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args recv_args;
  recv_args.struct_size =
      PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers_Args_STRUCT_SIZE;
  recv_args.extension_start = nullptr;
  recv_args.client = client_;
  recv_args.num_shapes = 1;
  std::vector<int64_t> shape = {2, 3};
  std::vector<size_t> shape_num_dims = {shape.size()};
  recv_args.shape_num_dims = shape_num_dims.data();
  std::vector<const int64_t*> num_dims = {shape.data()};
  recv_args.num_dims = num_dims.data();
  std::vector<PJRT_Buffer_Type> element_types = {
      PJRT_Buffer_Type::PJRT_Buffer_Type_F32};
  recv_args.element_types = element_types.data();
  std::vector<PJRT_Buffer_MemoryLayout*> layouts = {nullptr};
  recv_args.layouts = layouts.data();
  ASSERT_GE(GetClientDevices().size(), 2);
  recv_args.device = GetClientDevices()[1];

  // Create a notifier for the receiver to transmit the descriptor.
  auto [promise, future] = xla::Future<std::string>::MakePromise();
  xla::PjRtCrossHostRecvNotifier cpp_notifier =
      [promise = std::move(promise).ToShared()](
          absl::StatusOr<xla::PjRtCrossHostRecvState> recv_state) mutable {
        if (!recv_state.ok()) {
          promise->Set(recv_state.status());
          return;
        }
        std::string serialized_descriptor =
            recv_state->descriptors[0].serialized_descriptors.front();
        promise->Set(serialized_descriptor);
      };
  recv_args.notifier = pjrt::CppCrossHostRecvNotifierToC(api, cpp_notifier);
  std::vector<PJRT_Buffer*> temp_buffers(recv_args.num_shapes);
  recv_args.buffers = temp_buffers.data();
  cross_host_transfers_extension
      ->PJRT_Transfers_PJRT_Client_MakeCrossHostReceiveBuffers(&recv_args);

  std::vector<float> data = {1, 2, 3, 4, 5, 6};
  xla::Shape xla_shape =
      xla::ShapeUtil::MakeShape(xla::F32, /*dimensions=*/shape);
  auto [buffer, buffer_future] =
      create_buffer_from_data(data, xla_shape, GetClientDevices()[0]);
  TF_CHECK_OK(buffer_future.Await());

  // Send the buffer to the receiving device.
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized_descriptor, future.Await());
  PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args send_args;
  send_args.struct_size =
      PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice_Args_STRUCT_SIZE;
  send_args.extension_start = nullptr;
  send_args.buffer = buffer.get();
  send_args.serialized_descriptor = serialized_descriptor.c_str();
  send_args.serialized_descriptor_size = serialized_descriptor.size();
  cross_host_transfers_extension->PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice(
      &send_args);
  CHECK_EQ(recv_args.num_buffers, 1);
  auto recv_buffer = std::unique_ptr<PJRT_Buffer, PJRT_BufferDeleter>(
      recv_args.buffers[0], MakeBufferDeleter(api));
  TF_CHECK_OK(recv_buffer->buffer->GetReadyFuture().Await());

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> literal,
                          recv_buffer->buffer->ToLiteralSync());
  EXPECT_EQ(literal->element_count(), data.size());
  EXPECT_THAT(literal->data<float>(), ElementsAreArray(data));
}

}  // namespace
}  // namespace pjrt
