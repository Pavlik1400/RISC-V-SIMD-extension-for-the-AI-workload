#include "conf.h"
#if CFU_VERSION == 16

#include "cfu_utils.h"
#include "common.h"
#include "perf.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

namespace tflite {
namespace reference_integer_ops {

inline void ConvPerChannel_align2(const ConvParams& params,
                                  const int32_t* output_multiplier,
                                  const int32_t* output_shift,
                                  const RuntimeShape& input_shape,
                                  const int8_t* input_data,
                                  const RuntimeShape& filter_shape,
                                  const int8_t* filter_data,
                                  const RuntimeShape& bias_shape,
                                  const int32_t* bias_data,
                                  const RuntimeShape& output_shape,
                                  int8_t* output_data) {
  // Initialize cfu
  int32_t sum_at_once = cfu_op0(CFU_INITIALIZE, 0, 0);

  // Get parameters.
  const int32_t input_offset             = params.input_offset;  // r = s(q - Z)
  const int32_t output_offset            = params.output_offset;
  const int32_t output_activation_min    = -128;
  const int32_t output_activation_max    = 127;
  [[maybe_unused]] const int pad_width   = 3;
  const int output_depth                 = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_width                  = input_shape.Dims(2);
  const int filter_width                 = 8;
  [[maybe_unused]] const int input_depth = filter_shape.Dims(3);
  const int cfu_input_depth              = 4;  // smallest undividible cell in cfu bram is 4 bytes
  // const int output_width = output_shape.Dims(2);
  const int output_width = input_width;  // Since paddings = 'same'

  int write_at_once = 2;  // We write 2 values and 2 zeros at once

  int input_size         = input_depth * input_width;
  int filter_size        = input_depth * filter_width;
  int actual_filter_size = cfu_input_depth * filter_width;
  int actual_input_size  = cfu_input_depth * input_width;

  // Zero out filter buffer (4 values at a time)
  int filter_buffer_size = 8 * 64;
  for (int i = actual_filter_size; i < filter_buffer_size; i += 4) {
    cfu_op0(CFU_WRITE_FILTER_BUFFER, i, 0);
  }

  // zero out edge input values (cfu can access these values)
  int8_t min_input_offset         = static_cast<int8_t>(-input_offset);
  int8_t min_input_offset_arr2[2] = {min_input_offset, min_input_offset};
  int8_t min_input_offset_arr4[4] = {min_input_offset, min_input_offset, min_input_offset,
                                     min_input_offset};
  [[maybe_unused]] int16_t min_input_offset2 = *reinterpret_cast<int16_t*>(min_input_offset_arr2);
  int32_t min_input_offset4                  = *reinterpret_cast<int32_t*>(min_input_offset_arr4);

  for (int i = 0; i < sum_at_once; i += 4) {
    cfu_op0(CFU_WRITE_INPUT_BUFFER, i + actual_input_size, min_input_offset4);
  }

  // Fill input buffer (4 values at a time)
  for (int inp_addr = 0; inp_addr < input_size; inp_addr += write_at_once) {
    // 16 bits (2 values) are 0, other 2 are from input buffer
    int32_t value         = 0;
    int16_t* value_16_ptr = reinterpret_cast<int16_t*>(&value);
    value_16_ptr[0]       = *reinterpret_cast<const int16_t*>(input_data + inp_addr);
    cfu_op0(CFU_WRITE_INPUT_BUFFER, inp_addr * 2, value);  // * 2 because of 2 additional values
  }

  // Write parameters to CFU
  cfu_op0(CFU_WRITE_INPUT_OFFSET, 0, input_offset);
  cfu_op0(CFU_WRITE_OUTPUT_OFFSET, 0, output_offset);
  cfu_op0(CFU_WRITE_OUTPUT_ACTIVATION_MIN, 0, output_activation_min);
  cfu_op0(CFU_WRITE_OUTPUT_ACTIVATION_MAX, 0, output_activation_max);
  cfu_op0(CFU_WRITE_INPUT_OUTPUT_WIDTH, 0, input_width);
  cfu_op0(CFU_WRITE_INPUT_DEPTH, 0, cfu_input_depth);
  cfu_op0(CFU_WRITE_INPUT_WIDTH, 0, input_width);

  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    // Quant parameters
    cfu_op0(CFU_WRITE_BIAS, 0, bias_data[out_channel]);
    cfu_op0(CFU_WRITE_OUTPUT_MULTIPLIER, 0, output_multiplier[out_channel]);
    cfu_op0(CFU_WRITE_OUTPUT_SHIFT, 0, output_shift[out_channel]);

    // Copy kernel
    int filter_addr_offset = out_channel * filter_size;
    for (int kernel_addr = 0; kernel_addr < filter_size; kernel_addr += write_at_once) {
      int32_t value         = 0;
      int16_t* value_16_ptr = reinterpret_cast<int16_t*>(&value);
      int addr              = filter_addr_offset + kernel_addr;
      value_16_ptr[0]       = *reinterpret_cast<const int16_t*>(filter_data + addr);
      cfu_op0(CFU_WRITE_FILTER_BUFFER, kernel_addr * 2, value);
    }

    // input depth == no async writing -> buffer size is 1 row smaller
    int start_input_x  = -pad_width;
    int start_filter_x = pad_width;

    for (int out_x = 0; out_x < output_width; ++out_x) {
      cfu_op0(CFU_WRITE_START_INPUT_X, 0, start_input_x < 0 ? 0 : start_input_x);
      cfu_op0(CFU_WRITE_START_FILTER_X, 0, start_filter_x > 0 ? start_filter_x : 0);

      cfu_op0(CFU_START_COMPUTATION, 0, 0);
      while (!cfu_op0(CFU_FINISHED, 0, 0)) {
      };

      --start_filter_x;
      ++start_input_x;

      int32_t acc       = cfu_op0(CFU_READ_ACCUMULATOR, 0, 0);
      int addr          = out_x * output_depth + out_channel;
      output_data[addr] = static_cast<int8_t>(acc);

      // static int i = 0;
      // if (out_channel == 0) {
      //   printf("i: %d, acc: %ld\n", i++, acc);
      // }
      // static int i = 0;
      // if (i < 8) {
      //   printf("i: %d, acc: %ld\n", i, acc);
      //   ++i;
      // } else {
      //   exit(1);
      // }
    }
  }
  // exit(1);
}

inline void ConvPerChannel_align4(const ConvParams& params,
                                  const int32_t* output_multiplier,
                                  const int32_t* output_shift,
                                  const RuntimeShape& input_shape,
                                  const int8_t* input_data,
                                  const RuntimeShape& filter_shape,
                                  const int8_t* filter_data,
                                  const RuntimeShape& bias_shape,
                                  const int32_t* bias_data,
                                  const RuntimeShape& output_shape,
                                  int8_t* output_data) {
  // Initialize cfu
  cfu_op0(CFU_INITIALIZE, 0, 0);

  // Get parameters.
  const int32_t input_offset           = params.input_offset;  // r = s(q - Z)
  const int32_t output_offset          = params.output_offset;
  const int32_t output_activation_min  = -128;
  const int32_t output_activation_max  = 127;
  [[maybe_unused]] const int pad_width = 3;
  const int output_depth               = MatchingDim(filter_shape, 0, output_shape, 3);
  const int input_width                = input_shape.Dims(2);
  const int filter_width               = 8;
  const int input_depth                = filter_shape.Dims(3);
  // const int output_width = output_shape.Dims(2);
  const int output_width = input_width;  // Since paddings = 'same'

  int write_at_once = 4;

  int input_size  = input_depth * input_width;
  int filter_size = input_depth * filter_width;

  // Zero out unused filter buffer values (4 values at a time)
  int filter_buffer_size = 8 * 64;
  for (int i = filter_size; i < filter_buffer_size; i += write_at_once) {
    cfu_op0(CFU_WRITE_FILTER_BUFFER, i, 0);
  }

  // Fill input buffer (4 values at a time)
  for (int inp_addr = 0; inp_addr < input_size; inp_addr += write_at_once) {
    cfu_op0(CFU_WRITE_INPUT_BUFFER, inp_addr,
            *reinterpret_cast<const int32_t*>(input_data + inp_addr));
  }

  // Write parameters to CFU
  cfu_op0(CFU_WRITE_INPUT_OFFSET, 0, input_offset);
  cfu_op0(CFU_WRITE_OUTPUT_OFFSET, 0, output_offset);
  cfu_op0(CFU_WRITE_OUTPUT_ACTIVATION_MIN, 0, output_activation_min);
  cfu_op0(CFU_WRITE_OUTPUT_ACTIVATION_MAX, 0, output_activation_max);
  cfu_op0(CFU_WRITE_INPUT_OUTPUT_WIDTH, 0, input_width);
  cfu_op0(CFU_WRITE_INPUT_DEPTH, 0, input_depth);
  cfu_op0(CFU_WRITE_INPUT_WIDTH, 0, input_width);

  for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
    // Quant parameters
    cfu_op0(CFU_WRITE_BIAS, 0, bias_data[out_channel]);
    cfu_op0(CFU_WRITE_OUTPUT_MULTIPLIER, 0, output_multiplier[out_channel]);
    cfu_op0(CFU_WRITE_OUTPUT_SHIFT, 0, output_shift[out_channel]);

    // Copy kernel
    int filter_addr_offset = out_channel * filter_size;
    for (int kernel_addr = 0; kernel_addr < filter_size; kernel_addr += write_at_once) {
      int addr             = filter_addr_offset + kernel_addr;
      int32_t filter_value = *reinterpret_cast<const int32_t*>(filter_data + addr);
      cfu_op0(CFU_WRITE_FILTER_BUFFER, kernel_addr, filter_value);
    }

    // input depth == no async writing -> buffer size is 1 row smaller
    int start_input_x  = -pad_width;
    int start_filter_x = pad_width;

    for (int out_x = 0; out_x < output_width; ++out_x) {
      cfu_op0(CFU_WRITE_START_INPUT_X, 0, start_input_x > 0 ? start_input_x : 0);
      cfu_op0(CFU_WRITE_START_FILTER_X, 0, start_filter_x > 0 ? start_filter_x : 0);

      cfu_op0(CFU_START_COMPUTATION, 0, 0);
      while (!cfu_op0(CFU_FINISHED, 0, 0)) {
      };

      --start_filter_x;
      ++start_input_x;

      int32_t acc       = cfu_op0(CFU_READ_ACCUMULATOR, 0, 0);
      int addr          = out_x * output_depth + out_channel;
      output_data[addr] = static_cast<int8_t>(acc);
    }
  }
}

inline void ConvPerChannel(const ConvParams& params,
                           const int32_t* output_multiplier,
                           const int32_t* output_shift,
                           const RuntimeShape& input_shape,
                           const int8_t* input_data,
                           const RuntimeShape& filter_shape,
                           const int8_t* filter_data,
                           const RuntimeShape& bias_shape,
                           const int32_t* bias_data,
                           const RuntimeShape& output_shape,
                           int8_t* output_data) {
  const int input_depth = filter_shape.Dims(3);
  if (input_depth % 4 == 0) {
    return ConvPerChannel_align4(params, output_multiplier, output_shift, input_shape, input_data,
                                 filter_shape, filter_data, bias_shape, bias_data, output_shape,
                                 output_data);
  }
  if (input_depth % 2 == 0) {
    return ConvPerChannel_align2(params, output_multiplier, output_shift, input_shape, input_data,
                                 filter_shape, filter_data, bias_shape, bias_data, output_shape,
                                 output_data);
  } else {
    printf("<Error> : Alignment 4 and 2 supported only");
    exit(1);
  }
}
}  // namespace reference_integer_ops
}  // namespace tflite
#endif