import tensorflow as tf


class ExtractTensorFeatures(tf.keras.layers.Layer):
    def __init__(self, model):
        self._model = model
        super(ExtractTensorFeatures, self).__init__()

    def call(self, inputs, **kwargs):
        return tf.map_fn(lambda frame_data: self._model(frame_data), inputs)
