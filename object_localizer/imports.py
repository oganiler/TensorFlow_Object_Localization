# imports.py - centralized lazy imports to avoid heavy startup
_cache = {}

def get_tf():
    if "tf" not in _cache:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=7500)]  # 7.5GB
            )
        _cache["tf"] = tf
    return _cache["tf"]

def get_vgg16():
    if "vgg16" not in _cache:
        from tensorflow.keras.applications.vgg16 import VGG16
        _cache["vgg16"] = VGG16
    return _cache["vgg16"]

def get_keras_layers():
    if "layers" not in _cache:
        from tensorflow.keras import layers
        _cache["layers"] = layers
    return _cache["layers"]

#binary_crossentropy is used in the custom loss function for stage 2, so we need to import it here
def get_binary_crossentropy():
    if "binary_crossentropy" not in _cache:
        from tensorflow.keras.losses import binary_crossentropy
        _cache["binary_crossentropy"] = binary_crossentropy
    return _cache["binary_crossentropy"]

def get_categorical_crossentropy():
    if "categorical_crossentropy" not in _cache:
        from tensorflow.keras.losses import categorical_crossentropy
        _cache["categorical_crossentropy"] = categorical_crossentropy
    return _cache["categorical_crossentropy"]

def get_sparse_categorical_crossentropy():
    if "sparse_categorical_crossentropy" not in _cache:
        from tensorflow.keras.losses import sparse_categorical_crossentropy
        _cache["sparse_categorical_crossentropy"] = sparse_categorical_crossentropy
    return _cache["sparse_categorical_crossentropy"]

def get_mean_squared_error():
    if "mean_squared_error" not in _cache:
        from tensorflow.keras.losses import mean_squared_error
        _cache["mean_squared_error"] = mean_squared_error
    return _cache["mean_squared_error"]

def get_plt():
    if "plt" not in _cache:
        from matplotlib import pyplot as plt
        _cache["plt"] = plt
    return _cache["plt"]

def get_np():
    if "np" not in _cache:
        import numpy as np
        _cache["np"] = np
    return _cache["np"]