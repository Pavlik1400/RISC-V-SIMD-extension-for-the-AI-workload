#include "models/cnn_1d_v012_small_radio_ml16b_q_aware/cnn_1d_v012_small_radio_ml16b_q_aware.h"

#include <math.h>
#include <stdio.h>

#include "menu.h"
#include "models/cnn_1d_v012_small_radio_ml16b_q_aware/cnn_1d_v012_small_radio_ml16b_q_aware_model.h"
#include "models/cnn_1d_v012_small_radio_ml16b_q_aware/cnn_1d_v012_small_radio_ml16b_q_aware_test_data.h"
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
static void cnn_1d_v012_small_radio_ml16b_q_aware_init(void) { 
    tflite_load_model(cnn_1d_v012_small_radio_ml16b_q_aware_model, cnn_1d_v012_small_radio_ml16b_q_aware_model_len); 
}

// Run classification, after input has been loaded
static int8_t *cnn_1d_v012_small_radio_ml16b_q_aware_classify() {
  printf("Running cnn_1d_v012_small_radio_ml16b_q_aware model classification\n");
  tflite_classify();

  // Process the inference results.
  int8_t* output = (int8_t*)tflite_get_output();
  return output;
}

#define NUM_CLASSES 10

/* Returns true if failed */
static bool perform_one_test(int8_t* input, int8_t* expected_output, int8_t epsilon) {
  bool failed = false;
  tflite_set_input(input);
  int8_t* output = cnn_1d_v012_small_radio_ml16b_q_aware_classify();
  for (size_t i = 0; i < NUM_CLASSES; ++i) {
    int8_t y_true = expected_output[i];
    int8_t y_pred = output[i];

    
    int8_t delta = CFU_MAX(y_true, y_pred) - CFU_MIN(y_true, y_pred);
    if (delta > epsilon) {
      printf(
          "*** cnn_1d_v012_small_radio_ml16b_q_aware test failed %d (actual) != %d (pred). "
          "Class=%u\n",
          y_true, y_pred, i);
    
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
  int8_t epsilon = 64;
  bool failed = false;

  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_8PSK,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_8PSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_AM_DSB,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_AM_DSB, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_BPSK,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_BPSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_CPFSK,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_CPFSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_GFSK,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_GFSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_PAM4,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_PAM4, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_QAM16,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_QAM16, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_QAM64,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_QAM64, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_QPSK,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_QPSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml16b_q_aware_WBFM,
    pred_cnn_1d_v012_small_radio_ml16b_q_aware_WBFM, 
    epsilon
  );    
  

  if (failed) {
    printf("FAIL cnn_1d_v012_small_radio_ml16b_q_aware tests failed\n");
  } else {
    printf("OK   cnn_1d_v012_small_radio_ml16b_q_aware tests passed\n");
  }
}


static struct Menu MENU = {
    "Tests for cnn_1d_v012_small_radio_ml16b_q_aware model",
    "sine",
    {
        MENU_ITEM('g', "Run cnn_1d_v012_small_radio_ml16b_q_aware tests (check for expected outputs)", do_tests),
        MENU_END,
    },
};

// For integration into menu system
void cnn_1d_v012_small_radio_ml16b_q_aware_menu() {
  cnn_1d_v012_small_radio_ml16b_q_aware_init();
  menu_run(&MENU);
}