# imports.py - centralized lazy imports to avoid heavy startup
_cache = {}

def get_tf():
    if "tf" not in _cache:
        import tensorflow as tf
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