import tensorflow as tf

from base.base_dataset_processor import BaseDatasetProcessor
from configs.dataset.modality import DatasetFeaturesSet


class MultimodalDatasetFeaturesProcessor(BaseDatasetProcessor):

    def __init__(self, modalities_list):
        self._modalities_list = modalities_list

    def pre_process(self, dataset: tf.data.Dataset, parallel_calls: int):
        dataset = dataset.map(self._extract_specified_modalities_and_ensure_shape, num_parallel_calls=parallel_calls)
        dataset = dataset.map(self.concat_with_labels, num_parallel_calls=parallel_calls)
        return dataset

    @tf.function
    def concat_with_labels(self, example: tf.train.Example):
        inputs = []
        for modality in self._modalities_list:
            inputs.append(example[str(modality.name)])

        return tuple(inputs), example[DatasetFeaturesSet.CLASS.name]

    @tf.function
    def _extract_specified_modalities_and_ensure_shape(self, example: tf.Tensor) -> dict:
        tf_features_dict = {}
        for modality in self._modalities_list:
            data = example[str(modality.name)]
            tf_features_dict[str(modality.name)] = tf.ensure_shape(data, modality.config.shape)

        tf_features_dict[DatasetFeaturesSet.CLASS.name] = example[DatasetFeaturesSet.CLASS.name]
        return tf_features_dict

    @tf.function
    def filter_nan(self, input_, output):
        return not tf.reduce_any(tf.math.is_nan(input_)) and not tf.reduce_any(tf.math.is_nan(output))
