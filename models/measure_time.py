import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import time
import numpy as np
from deployment_tools import predict_tflite, to_tf_lite
from datasets.fabric import make_sigmod_ds, DatasetName



# tf.config.set_visible_devices([], 'GPU')

model_cnn_path = "experiments/cnn_1d_v013_small_radio_ml16b_results/model_original"
model_encoder_path = "experiments/enc_v3_small_radio_ml16b_normalized_results/model_original"

model_path = model_encoder_path
# model_path = model_cnn_path
model = to_tf_lite(model_path)


radioml2016b_path = "data/radioml_2016/RML2016.10b.dat"
radioml_ds = make_sigmod_ds(DatasetName.RADIOML_2016)
expand2d = model_path == model_cnn_path
radioml_ds.load(radioml2016b_path, expand2d=expand2d)
data = radioml_ds.get_data()

# data = data[:40_000]
data = data[:1_000]

warm_up = 5
n_iterations = 10
batch_size = 1

for i in range(warm_up):
    print(f"Warm up: {i}")
    # pred = model.predict(data, batch_size=batch_size, verbose=False)
    for k in range(len(data)):
        # pred = model.predict(np.expand_dims(data[k], 1), 1, verbose=False)
        pred = predict_tflite(model, np.expand_dims(data[k], 0))
        # pred = predict_tflite(model, data[k])

start = time.time()
for i in range(n_iterations):
    print(f"Run: {i}")
    # pred = model.predict(data, batch_size=batch_size, verbose=False)
    for k in range(len(data)):
        # pred = model.predict(np.expand_dims(data[k], 1), 1, verbose=False)
        pred = predict_tflite(model, np.expand_dims(data[k], 0))
        # pred = predict_tflite(model, data[k])

end = time.time()

elapsed = end - start
print(f"Time elapsed: {elapsed}s")
print(f"Time per frame: {(elapsed / (len(data) * n_iterations)) * 1000}ms")

