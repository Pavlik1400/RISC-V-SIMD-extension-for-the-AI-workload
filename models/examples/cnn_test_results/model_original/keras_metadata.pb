
��root"_tf_keras_network*��{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "CNN0_", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN0_", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_0_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_0_", "inbound_nodes": [[["CNN0_", 0, 0, {}]]]}, {"class_name": "CustomBatchNorm", "config": {"name": "BN0_", "trainable": true, "dtype": "float32"}, "name": "BN0_", "inbound_nodes": [[["MAX_POOL_0_", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "CNN_REL0_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL0_", "inbound_nodes": [[["BN0_", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "CNN1_", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN1_", "inbound_nodes": [[["CNN_REL0_", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_1_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_1_", "inbound_nodes": [[["CNN1_", 0, 0, {}]]]}, {"class_name": "CustomBatchNorm", "config": {"name": "BN1_", "trainable": true, "dtype": "float32"}, "name": "BN1_", "inbound_nodes": [[["MAX_POOL_1_", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "CNN_REL1_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL1_", "inbound_nodes": [[["BN1_", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "CNN2_", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN2_", "inbound_nodes": [[["CNN_REL1_", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_2_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_2_", "inbound_nodes": [[["CNN2_", 0, 0, {}]]]}, {"class_name": "CustomBatchNorm", "config": {"name": "BN2_", "trainable": true, "dtype": "float32"}, "name": "BN2_", "inbound_nodes": [[["MAX_POOL_2_", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "CNN_REL2_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL2_", "inbound_nodes": [[["BN2_", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "CNN3_", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN3_", "inbound_nodes": [[["CNN_REL2_", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_3_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_3_", "inbound_nodes": [[["CNN3_", 0, 0, {}]]]}, {"class_name": "CustomBatchNorm", "config": {"name": "BN3_", "trainable": true, "dtype": "float32"}, "name": "BN3_", "inbound_nodes": [[["MAX_POOL_3_", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "CNN_REL3_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL3_", "inbound_nodes": [[["BN3_", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "CNN4_", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN4_", "inbound_nodes": [[["CNN_REL3_", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_4_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_4_", "inbound_nodes": [[["CNN4_", 0, 0, {}]]]}, {"class_name": "CustomBatchNorm", "config": {"name": "BN4_", "trainable": true, "dtype": "float32"}, "name": "BN4_", "inbound_nodes": [[["MAX_POOL_4_", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "CNN_REL4_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL4_", "inbound_nodes": [[["BN4_", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "CNN5_", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN5_", "inbound_nodes": [[["CNN_REL4_", 0, 0, {}]]]}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_5_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_5_", "inbound_nodes": [[["CNN5_", 0, 0, {}]]]}, {"class_name": "CustomBatchNorm", "config": {"name": "BN5_", "trainable": true, "dtype": "float32"}, "name": "BN5_", "inbound_nodes": [[["MAX_POOL_5_", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "CNN_REL5_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL5_", "inbound_nodes": [[["BN5_", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "AVG1_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [32]}, "pool_size": {"class_name": "__tuple__", "items": [32]}, "padding": "valid", "data_format": "channels_last"}, "name": "AVG1_", "inbound_nodes": [[["CNN_REL5_", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "FLT1_", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "FLT1_", "inbound_nodes": [[["AVG1_", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "FC_1_", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1_", "inbound_nodes": [[["FLT1_", 0, 0, {}]]]}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}, "name": "softmax", "inbound_nodes": [[["FC_1_", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["softmax", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["flatten", 0, 0]]}, "shared_object_id": 44, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 128, 2]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 2]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 2]}, "float32", "input_1"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 128, 2]}, "float32", "input_1"]}, "keras_version": "2.12.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv1D", "config": {"name": "CNN0_", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN0_", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_0_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_0_", "inbound_nodes": [[["CNN0_", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "CustomBatchNorm", "config": {"name": "BN0_", "trainable": true, "dtype": "float32"}, "name": "BN0_", "inbound_nodes": [[["MAX_POOL_0_", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "ReLU", "config": {"name": "CNN_REL0_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL0_", "inbound_nodes": [[["BN0_", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Conv1D", "config": {"name": "CNN1_", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN1_", "inbound_nodes": [[["CNN_REL0_", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_1_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_1_", "inbound_nodes": [[["CNN1_", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "CustomBatchNorm", "config": {"name": "BN1_", "trainable": true, "dtype": "float32"}, "name": "BN1_", "inbound_nodes": [[["MAX_POOL_1_", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "ReLU", "config": {"name": "CNN_REL1_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL1_", "inbound_nodes": [[["BN1_", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Conv1D", "config": {"name": "CNN2_", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN2_", "inbound_nodes": [[["CNN_REL1_", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_2_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_2_", "inbound_nodes": [[["CNN2_", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "CustomBatchNorm", "config": {"name": "BN2_", "trainable": true, "dtype": "float32"}, "name": "BN2_", "inbound_nodes": [[["MAX_POOL_2_", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "ReLU", "config": {"name": "CNN_REL2_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL2_", "inbound_nodes": [[["BN2_", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv1D", "config": {"name": "CNN3_", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN3_", "inbound_nodes": [[["CNN_REL2_", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_3_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_3_", "inbound_nodes": [[["CNN3_", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "CustomBatchNorm", "config": {"name": "BN3_", "trainable": true, "dtype": "float32"}, "name": "BN3_", "inbound_nodes": [[["MAX_POOL_3_", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "ReLU", "config": {"name": "CNN_REL3_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL3_", "inbound_nodes": [[["BN3_", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "Conv1D", "config": {"name": "CNN4_", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN4_", "inbound_nodes": [[["CNN_REL3_", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_4_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_4_", "inbound_nodes": [[["CNN4_", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "CustomBatchNorm", "config": {"name": "BN4_", "trainable": true, "dtype": "float32"}, "name": "BN4_", "inbound_nodes": [[["MAX_POOL_4_", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "ReLU", "config": {"name": "CNN_REL4_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL4_", "inbound_nodes": [[["BN4_", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Conv1D", "config": {"name": "CNN5_", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CNN5_", "inbound_nodes": [[["CNN_REL4_", 0, 0, {}]]], "shared_object_id": 33}, {"class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_5_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "name": "MAX_POOL_5_", "inbound_nodes": [[["CNN5_", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "CustomBatchNorm", "config": {"name": "BN5_", "trainable": true, "dtype": "float32"}, "name": "BN5_", "inbound_nodes": [[["MAX_POOL_5_", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "ReLU", "config": {"name": "CNN_REL5_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "name": "CNN_REL5_", "inbound_nodes": [[["BN5_", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "AveragePooling1D", "config": {"name": "AVG1_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [32]}, "pool_size": {"class_name": "__tuple__", "items": [32]}, "padding": "valid", "data_format": "channels_last"}, "name": "AVG1_", "inbound_nodes": [[["CNN_REL5_", 0, 0, {}]]], "shared_object_id": 37}, {"class_name": "Flatten", "config": {"name": "FLT1_", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "FLT1_", "inbound_nodes": [[["AVG1_", 0, 0, {}]]], "shared_object_id": 38}, {"class_name": "Dense", "config": {"name": "FC_1_", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "FC_1_", "inbound_nodes": [[["FLT1_", 0, 0, {}]]], "shared_object_id": 41}, {"class_name": "Softmax", "config": {"name": "softmax", "trainable": true, "dtype": "float32", "axis": -1}, "name": "softmax", "inbound_nodes": [[["FC_1_", 0, 0, {}]]], "shared_object_id": 42}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["softmax", 0, 0, {}]]], "shared_object_id": 43}], "input_layers": [["input_1", 0, 0]], "output_layers": [["flatten", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false, "ignore_class": null}, "shared_object_id": 46}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 47}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": true, "is_legacy_optimizer": false, "learning_rate": 0.0010000000474974513, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}2
�
root.layer_with_weights-0"_tf_keras_layer*�	{"name": "CNN0_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv1D", "config": {"name": "CNN0_", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 2}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 2]}}2
�root.layer-2"_tf_keras_layer*�{"name": "MAX_POOL_0_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_0_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["CNN0_", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 32]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "BN0_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CustomBatchNorm", "config": {"name": "BN0_", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["MAX_POOL_0_", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 32]}}2
�root.layer-4"_tf_keras_layer*�{"name": "CNN_REL0_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "ReLU", "config": {"name": "CNN_REL0_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["BN0_", 0, 0, {}]]], "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 32]}}2
�
root.layer_with_weights-2"_tf_keras_layer*�	{"name": "CNN1_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv1D", "config": {"name": "CNN1_", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["CNN_REL0_", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 32]}}2
�root.layer-6"_tf_keras_layer*�{"name": "MAX_POOL_1_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "MaxPooling1D", "config": {"name": "MAX_POOL_1_", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [1]}, "pool_size": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last"}, "inbound_nodes": [[["CNN1_", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 48]}}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "BN1_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CustomBatchNorm", "config": {"name": "BN1_", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["MAX_POOL_1_", 0, 0, {}]]], "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 48]}}2
�	root.layer-8"_tf_keras_layer*�{"name": "CNN_REL1_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "ReLU", "config": {"name": "CNN_REL1_", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}, "inbound_nodes": [[["BN1_", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 48]}}2
�

root.layer_with_weights-4"_tf_keras_layer*�	{"name": "CNN2_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv1D", "config": {"name": "CNN2_", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["CNN_REL1_", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 48}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 48]}}2
�
�root.layer_with_weights-5"_tf_keras_layer*�{"name": "BN2_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CustomBatchNorm", "config": {"name": "BN2_", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["MAX_POOL_2_", 0, 0, {}]]], "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64]}}2
�
�
root.layer_with_weights-6"_tf_keras_layer*�	{"name": "CNN3_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv1D", "config": {"name": "CNN3_", "trainable": true, "dtype": "float32", "filters": 96, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["CNN_REL2_", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64]}}2
�
�root.layer_with_weights-7"_tf_keras_layer*�{"name": "BN3_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CustomBatchNorm", "config": {"name": "BN3_", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["MAX_POOL_3_", 0, 0, {}]]], "shared_object_id": 23, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 96]}}2
�
�
root.layer_with_weights-8"_tf_keras_layer*�	{"name": "CNN4_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv1D", "config": {"name": "CNN4_", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["CNN_REL3_", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 96}}, "shared_object_id": 56}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 96]}}2
�
�root.layer_with_weights-9"_tf_keras_layer*�{"name": "BN4_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CustomBatchNorm", "config": {"name": "BN4_", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["MAX_POOL_4_", 0, 0, {}]]], "shared_object_id": 29, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128]}}2
�
�
root.layer_with_weights-10"_tf_keras_layer*�	{"name": "CNN5_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Conv1D", "config": {"name": "CNN5_", "trainable": true, "dtype": "float32", "filters": 192, "kernel_size": {"class_name": "__tuple__", "items": [8]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["CNN_REL4_", 0, 0, {}]]], "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 128}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 128]}}2
�
�root.layer_with_weights-11"_tf_keras_layer*�{"name": "BN5_", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CustomBatchNorm", "config": {"name": "BN5_", "trainable": true, "dtype": "float32"}, "inbound_nodes": [[["MAX_POOL_5_", 0, 0, {}]]], "shared_object_id": 35, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 192]}}2
�
�
�
�root.layer_with_weights-12"_tf_keras_layer*�{"name": "FC_1_", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "FC_1_", "trainable": true, "dtype": "float32", "units": 11, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 39}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 40}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["FLT1_", 0, 0, {}]]], "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 192}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192]}}2
�
�
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 64}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 47}2