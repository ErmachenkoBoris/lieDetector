import tensorflow as tf

from configs.dataset.modality import DatasetFeaturesSet
from utils.dirs import get_files_from_dir
from sklearn.model_selection import train_test_split


class DataManager:

    def __init__(self,
                 tf_record_path: str,
                 repeat: int = None,
                 batch_size: int = 1,
                 use_cache: bool = True,
                 use_prefetch: bool = True):
        self._tf_record_path = tf_record_path
        self._use_cache = use_cache
        self._batch_size = batch_size

        self._use_prefetch = use_prefetch
        self._repeat = repeat

        files = get_files_from_dir(tf_record_path)
        print("Dataset files: {}".format(files))
        self.PARALLEL_CALLS = tf.data.experimental.AUTOTUNE

        self.train_files, self.val_files = train_test_split(files, test_size=0.3)

        print("Train files size: {}".format(len(self.train_files)))
        print("Valid files size: {}".format(len(self.val_files)))

        self._val_ds = tf.data.TFRecordDataset(self.val_files)
        self._train_ds = tf.data.TFRecordDataset(self.train_files)

        self._feature_description = {
            DatasetFeaturesSet.OPENSMILE_ComParE_2016.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.AUDIO.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_FACE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_SCENE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.PULSE.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.CLASS.name: tf.io.FixedLenFeature([], tf.string),
            DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name: tf.io.FixedLenFeature([], tf.string),
        }

        self._val_ds = self.decode_data(self._val_ds)
        self._train_ds = self.decode_data(self._train_ds)

        if self._use_cache:
            self._val_ds = self._val_ds.cache()
            self._train_ds = self._train_ds.cache()

    def build_training_dataset(self, dataset_processor) -> tf.data.Dataset:
        return self._preprocess_dataset(dataset_processor, self._train_ds)

    def build_validation_dataset(self, dataset_processor) -> tf.data.Dataset:
        return self._preprocess_dataset(dataset_processor, self._val_ds)

    def _preprocess_dataset(self, dataset_processor, ds: tf.data.Dataset, ) -> tf.data.Dataset:
        ds = dataset_processor.pre_process(ds, self.PARALLEL_CALLS)

        ds = ds.batch(self._batch_size)
        #
        # if self._repeat:
        #     ds = ds.repeat(self._repeat)
        # else:
        #     ds = ds.repeat()

        if self._use_prefetch:
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def decode_data(self, dataset):
        dataset = dataset.map(self.map_record_to_dictionary_of_tensors,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    @tf.function
    def map_record_to_dictionary_of_tensors(self, serialized_example: tf.train.Example) -> dict:
        return self._decode_example(serialized_example)

    @tf.function
    def _decode_example(self, serialized_example: tf.Tensor) -> dict:
        example = tf.io.parse_single_example(serialized_example, self._feature_description)

        opensmile_compare_2016_features = example[DatasetFeaturesSet.OPENSMILE_ComParE_2016.name]
        opensmile_compare_2016_features = tf.io.parse_tensor(opensmile_compare_2016_features, tf.float32)
        opensmile_compare_2016_features = tf.squeeze(opensmile_compare_2016_features)
        opensmile_compare_2016_features = tf.cast(opensmile_compare_2016_features, tf.float32)

        audio_features = example[DatasetFeaturesSet.AUDIO.name]
        audio_features = tf.io.parse_tensor(audio_features, tf.float32)
        audio_features = tf.cast(audio_features, tf.float32)

        video_face_features = example[DatasetFeaturesSet.VIDEO_FACE.name]
        video_face_features = tf.io.parse_tensor(video_face_features, tf.float32)
        video_face_features = tf.cast(video_face_features, tf.float32)

        video_scene_features = example[DatasetFeaturesSet.VIDEO_SCENE.name]
        video_scene_features = tf.io.parse_tensor(video_scene_features, tf.float32)
        video_scene_features = tf.cast(video_scene_features, tf.float32)

        pulse_features = example[DatasetFeaturesSet.PULSE.name]
        pulse_features = tf.io.parse_tensor(pulse_features, tf.float32)
        pulse_features = tf.cast(pulse_features, tf.float32)
        pulse_features = tf.expand_dims(pulse_features, -1)

        video_scene_r2plus1_features = example[DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name]
        video_scene_r2plus1_features = tf.io.parse_tensor(video_scene_r2plus1_features, tf.float32)
        video_scene_r2plus1_features = tf.cast(video_scene_r2plus1_features, tf.float32)

        clazz = tf.io.parse_tensor(example[DatasetFeaturesSet.CLASS.name], tf.float32)
        clazz = tf.cast(clazz, dtype=tf.float32)
        clazz = tf.ensure_shape(clazz, 1)

        return {
            DatasetFeaturesSet.OPENSMILE_ComParE_2016.name: opensmile_compare_2016_features,
            DatasetFeaturesSet.AUDIO.name: audio_features,
            DatasetFeaturesSet.VIDEO_FACE.name: video_face_features,
            DatasetFeaturesSet.VIDEO_SCENE.name: video_scene_features,
            DatasetFeaturesSet.PULSE.name: pulse_features,
            DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES.name: video_scene_r2plus1_features,
            DatasetFeaturesSet.CLASS.name: clazz
        }
