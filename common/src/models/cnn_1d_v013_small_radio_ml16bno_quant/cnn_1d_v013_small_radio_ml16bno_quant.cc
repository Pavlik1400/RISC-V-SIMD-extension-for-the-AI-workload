#include "models/cnn_1d_v013_small_radio_ml16bno_quant/cnn_1d_v013_small_radio_ml16bno_quant.h"

#include <math.h>
#include <stdio.h>

#include "menu.h"
#include "models/cnn_1d_v013_small_radio_ml16bno_quant/cnn_1d_v013_small_radio_ml16bno_quant_model.h"
#include "models/cnn_1d_v013_small_radio_ml16bno_quant/cnn_1d_v013_small_radio_ml16bno_quant_test_data.h"
#include "tensorflow/lite/micro/examples/person_detection/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "playground_util/models_utils.h"
#include "tflite.h"

extern "C" {
#include "fb_util.h"
};

#ifndef CFU_MAX
#define CFU_MAX(x, y) (((x) >= (y)) ? (x) : (y))
#endif  // CFU_MAX

#ifndef CFU_MIN
#define CFU_MIN(x, y) (((x) <= (y)) ? (x) : (y))
#endif  // CFU_MIN

// This method creates interpreter, arena, loads model to memory
static void cnn_1d_v013_small_radio_ml16bno_quant_init(void) { 
    tflite_load_model(cnn_1d_v013_small_radio_ml16bno_quant_model, cnn_1d_v013_small_radio_ml16bno_quant_model_len); 
}

// Run classification, after input has been loaded
static float *cnn_1d_v013_small_radio_ml16bno_quant_classify() {
  printf("Running cnn_1d_v013_small_radio_ml16bno_quant model classification\n");
  tflite_classify();

  // Process the inference results.
  float* output = (float*)tflite_get_output();
  return output;
}

#define NUM_CLASSES 10

/* Returns true if failed */
static bool perform_one_test(float* input, float* expected_output, float epsilon) {
  bool failed = false;
  tflite_set_input(input);
  float* output = cnn_1d_v013_small_radio_ml16bno_quant_classify();
  for (size_t i = 0; i < NUM_CLASSES; ++i) {
    float y_true = expected_output[i];
    float y_pred = output[i];

    
    float delta = CFU_MAX(y_true, y_pred) - CFU_MIN(y_true, y_pred);
    int* y_true_i32_ptr = (int*)(&y_true);
    int* y_pred_i32_ptr = (int*)(&y_pred);
    if (delta > epsilon) {
      printf(
          "*** cnn_1d_v013_small_radio_ml16bno_quant test failed %d (actual) != %d (pred). "
          "Class=%u\n",
          *y_true_i32_ptr, *y_pred_i32_ptr, i);
    
      failed = true;
    } else {
      // printf(
      //     "+++ Signal modulation 1 test success %d (actual) != %d (pred). "
      //     "Class=%u\n",
      //     *y_true_u32_ptr, *y_pred_u32_ptr, i);
    }
  }
  return failed;
}

static void do_tests() {
  float epsilon = 0.01;
  bool failed = false;

  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_8PSK,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_8PSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_AM_DSB,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_AM_DSB, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_BPSK,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_BPSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_CPFSK,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_CPFSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_GFSK,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_GFSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_PAM4,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_PAM4, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_QAM16,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_QAM16, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_QAM64,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_QAM64, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_QPSK,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_QPSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v013_small_radio_ml16bno_quant_WBFM,
    pred_cnn_1d_v013_small_radio_ml16bno_quant_WBFM, 
    epsilon
  );    
  

  if (failed) {
    printf("FAIL cnn_1d_v013_small_radio_ml16bno_quant tests failed\n");
  } else {
    printf("OK   cnn_1d_v013_small_radio_ml16bno_quant tests passed\n");
  }
}


static struct Menu MENU = {
    "Tests for cnn_1d_v013_small_radio_ml16bno_quant model",
    "sine",
    {
        MENU_ITEM('g', "Run cnn_1d_v013_small_radio_ml16bno_quant tests (check for expected outputs)", do_tests),
        MENU_END,
    },
};

// For integration into menu system
void cnn_1d_v013_small_radio_ml16bno_quant_menu() {
  cnn_1d_v013_small_radio_ml16bno_quant_init();
  menu_run(&MENU);
}