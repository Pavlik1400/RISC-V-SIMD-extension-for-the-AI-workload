#pragma once

#include "software_cfu_common.hh"

// Basically the whole conv.h moved to the cfu
#define MAX_OUTPUT_CHANNELS 192
uint32_t software_cfu_v2(int funct3, int funct7, uint32_t rs1, uint32_t rs2) {
  if (funct3 != 0) {
    return 0;
  }

  // Buffers
  static int8_t input_buffer[MAX_INPUT_SIZE * MAX_INPUT_CHANNELS];
  static int8_t kernel_weights_buffer[KERNEL_LENGTH * MAX_INPUT_CHANNELS * MAX_OUTPUT_CHANNELS];
  static int8_t output_buffer[MAX_INPUT_SIZE * MAX_OUTPUT_CHANNELS];
  static int32_t bias_buffer[MAX_OUTPUT_CHANNELS];
  static int32_t output_multiplier_buffer[MAX_OUTPUT_CHANNELS];
  static int32_t output_shift_buffer[MAX_OUTPUT_CHANNELS];

  // Static variables
  // static int8_t bias          = 0;
  static int32_t input_offset = 0;

  static int32_t output_offset         = 0;
  static int32_t output_activation_min = 0;
  static int32_t output_activation_max = 0;

  static int output_depth       = 0;
  static int input_output_width = 0;
  static int input_depth        = 0;

  // Other variables
  uint32_t address = rs1;
  uint32_t value   = rs2;

  // State machine
  switch (funct7) {
    case 00:  // Zero out buffers
      for (uint32_t out_idx = 0; out_idx < MAX_INPUT_SIZE * MAX_OUTPUT_CHANNELS; ++out_idx) {
        output_buffer[out_idx] = 0;
      }

      for (uint32_t in_idx = 0; in_idx < MAX_INPUT_SIZE * MAX_INPUT_CHANNELS; ++in_idx) {
        input_buffer[in_idx] = 0;
      }

      for (uint32_t kernel_idx = 0;
           kernel_idx < KERNEL_LENGTH * MAX_INPUT_CHANNELS * MAX_OUTPUT_CHANNELS; ++kernel_idx) {
        kernel_weights_buffer[kernel_idx] = 0;
      }

      for (uint32_t output_channel = 0; output_channel < MAX_OUTPUT_CHANNELS; ++output_channel) {
        bias_buffer[output_activation_max]              = 0;
        output_multiplier_buffer[output_activation_max] = 0;
        output_shift_buffer[output_activation_max]      = 0;
      }
      return 0;

    case 10:  // Write to input buffer
      input_buffer[address] = (int8_t)value;
      return 0;
    case 20:  // Write to kernel weights buffer
      kernel_weights_buffer[address] = (int8_t)value;
      return 0;
    case 30:  // Read output buffer
      return output_buffer[address];
    case 40:  // Read input buffer
      return input_buffer[address];
    case 50:  // Read kernel weights buffer
      return kernel_weights_buffer[address];
    // case 60:  // Write bias
    //   bias = (int8_t)rs1;
    //   return bias;
    case 70:  // Write input offset
      input_offset = (int32_t)rs1;
      return input_offset;
    case 80:  // Write output_offset
      output_offset = (int32_t)rs1;
      return output_offset;
    case 90:  // Write output_activation_min
      output_activation_min = (int32_t)rs1;
      return output_activation_min;
    case 100:  // Write output_activation_max
      output_activation_max = (int32_t)rs1;
      return output_activation_max;
    case 110:  // Write output_depth
      output_depth = (int)rs1;
      return output_depth;
    case 120:  // Write input_output_width
      input_output_width = (int)rs1;
      return input_output_width;
    case 130:  // Write input_depth
      input_depth = (int)rs1;
      return input_depth;
    case 140:  // Write bias_buffer
      bias_buffer[address] = (int32_t)value;
      return 0;
    case 150:  // Write output_multiplier_buffer
      output_multiplier_buffer[address] = (int32_t)value;
      return 0;
    case 160:  // Write output_shift_buffer
      output_shift_buffer[address] = (int32_t)value;
      return 0;

    case 170:  // Start computation
      for (int out_x = 0; out_x < input_output_width; ++out_x) {
        const int in_x_origin = out_x - (PADDING - 1);
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          int32_t acc = 0;
          for (int filter_x = 0; filter_x < 8; ++filter_x) {
            const int in_x = in_x_origin + filter_x;

            // Zero padding by omitting the areas outside the image.
            const bool is_point_inside_image = (in_x >= 0) && (in_x < input_output_width);
            if (!is_point_inside_image) {
              continue;
            }

            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
              int32_t input_val  = input_buffer[in_x * input_depth + in_channel];
              int32_t filter_val = kernel_weights_buffer[out_channel * (8 * input_depth) +
                                                         filter_x * input_depth + in_channel];

              acc += filter_val * (input_val + input_offset);
            }
          }

          acc += bias_buffer[out_channel];
          acc = multiply_by_quant_mult(acc, (int32_t)output_multiplier_buffer[out_channel],
                                       (int32_t)output_shift_buffer[out_channel]);
          acc += output_offset;
          acc                                               = std::max(acc, output_activation_min);
          acc                                               = std::min(acc, output_activation_max);
          output_buffer[out_x * output_depth + out_channel] = static_cast<int8_t>(acc);
        }
      }
      return 0;
    default:
      return 0;
  }
}