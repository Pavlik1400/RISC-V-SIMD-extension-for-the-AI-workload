#include "models/cnn_1d_v012_small_radio_ml18/cnn_1d_v012_small_radio_ml18.h"

#include <math.h>
#include <stdio.h>

#include "menu.h"
#include "models/cnn_1d_v012_small_radio_ml18/cnn_1d_v012_small_radio_ml18_model.h"
#include "models/cnn_1d_v012_small_radio_ml18/cnn_1d_v012_small_radio_ml18_test_data.h"
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
static void cnn_1d_v012_small_radio_ml18_init(void) { 
    tflite_load_model(cnn_1d_v012_small_radio_ml18_model, cnn_1d_v012_small_radio_ml18_model_len); 
}

// Run classification, after input has been loaded
static int8_t *cnn_1d_v012_small_radio_ml18_classify() {
  printf("Running cnn_1d_v012_small_radio_ml18 model classification\n");
  tflite_classify();

  // Process the inference results.
  int8_t* output = (int8_t*)tflite_get_output();
  return output;
}

#define NUM_CLASSES 24

/* Returns true if failed */
static bool perform_one_test(int8_t* input, int8_t* expected_output, int8_t epsilon) {
  bool failed = false;
  tflite_set_input(input);
  int8_t* output = cnn_1d_v012_small_radio_ml18_classify();
  for (size_t i = 0; i < NUM_CLASSES; ++i) {
    int8_t y_true = expected_output[i];
    int8_t y_pred = output[i];

    
    int8_t delta = CFU_MAX(y_true, y_pred) - CFU_MIN(y_true, y_pred);
    if (delta > epsilon) {
      printf(
          "*** cnn_1d_v012_small_radio_ml18 test failed %d (actual) != %d (pred). "
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
  int8_t epsilon = 100;
  bool failed = false;

  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_32PSK,
    pred_cnn_1d_v012_small_radio_ml18_32PSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_16APSK,
    pred_cnn_1d_v012_small_radio_ml18_16APSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_32QAM,
    pred_cnn_1d_v012_small_radio_ml18_32QAM, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_FM,
    pred_cnn_1d_v012_small_radio_ml18_FM, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_GMSK,
    pred_cnn_1d_v012_small_radio_ml18_GMSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_32APSK,
    pred_cnn_1d_v012_small_radio_ml18_32APSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_OQPSK,
    pred_cnn_1d_v012_small_radio_ml18_OQPSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_8ASK,
    pred_cnn_1d_v012_small_radio_ml18_8ASK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_BPSK,
    pred_cnn_1d_v012_small_radio_ml18_BPSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_8PSK,
    pred_cnn_1d_v012_small_radio_ml18_8PSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_AM_SSB_SC,
    pred_cnn_1d_v012_small_radio_ml18_AM_SSB_SC, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_4ASK,
    pred_cnn_1d_v012_small_radio_ml18_4ASK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_16PSK,
    pred_cnn_1d_v012_small_radio_ml18_16PSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_64APSK,
    pred_cnn_1d_v012_small_radio_ml18_64APSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_128QAM,
    pred_cnn_1d_v012_small_radio_ml18_128QAM, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_128APSK,
    pred_cnn_1d_v012_small_radio_ml18_128APSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_AM_DSB_SC,
    pred_cnn_1d_v012_small_radio_ml18_AM_DSB_SC, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_AM_SSB_WC,
    pred_cnn_1d_v012_small_radio_ml18_AM_SSB_WC, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_64QAM,
    pred_cnn_1d_v012_small_radio_ml18_64QAM, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_QPSK,
    pred_cnn_1d_v012_small_radio_ml18_QPSK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_256QAM,
    pred_cnn_1d_v012_small_radio_ml18_256QAM, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_AM_DSB_WC,
    pred_cnn_1d_v012_small_radio_ml18_AM_DSB_WC, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_OOK,
    pred_cnn_1d_v012_small_radio_ml18_OOK, 
    epsilon
  );    
  
  failed = failed || perform_one_test(
    test_data_cnn_1d_v012_small_radio_ml18_16QAM,
    pred_cnn_1d_v012_small_radio_ml18_16QAM, 
    epsilon
  );    
  

  if (failed) {
    printf("FAIL cnn_1d_v012_small_radio_ml18 tests failed\n");
  } else {
    printf("OK   cnn_1d_v012_small_radio_ml18 tests passed\n");
  }
}


static struct Menu MENU = {
    "Tests for cnn_1d_v012_small_radio_ml18 model",
    "sine",
    {
        MENU_ITEM('g', "Run cnn_1d_v012_small_radio_ml18 tests (check for expected outputs)", do_tests),
        MENU_END,
    },
};

// For integration into menu system
void cnn_1d_v012_small_radio_ml18_menu() {
  cnn_1d_v012_small_radio_ml18_init();
  menu_run(&MENU);
}