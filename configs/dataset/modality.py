from enum import Enum
import tensorflow as tf
from abc import ABC

from models.exrtactors_type import AudioFeatureExtractor, VideoFeatureExtractor


class TimeDependentModality(ABC):
    WINDOW_STEP_IN_SECS = 1
    WINDOW_WIDTH_IN_SECS = 5


class VideoModalityConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.FRAMES_PERIOD = 5
        self.SHAPE = 224
        self.FPS = 32
        self.FILE_EXT = '.mp4'


class PulseConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.FPS = 50
        self.SHAPE = 224
        self.FILE_EXT = '.mp4'


class VideoSceneModalityConfig(VideoModalityConfig):
    def __init__(self):
        super().__init__()
        self.FILE_EXT = '.mp4'


class AudioModalityConfig(TimeDependentModality):
    def __init__(self):
        super().__init__()
        self.SR = 48000
        self.FILE_EXT = '.wav'


class Modality(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config):
        self.config = config

    AUDIO = AudioModalityConfig()
    VIDEO_FACE = VideoModalityConfig()
    VIDEO_SCENE = VideoSceneModalityConfig()
    PULSE = PulseConfig()


class ByteEncoder:
    def transform(self, feature):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))


class IntEncoder:
    def transform(self, feature):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))


class TensorEncoder(ByteEncoder):
    def transform(self, feature):
        feature = tf.io.serialize_tensor(tf.constant(feature)).numpy()
        return super().transform(feature)


class FeaturesSetConfig:
    def __init__(self, shape, extractor, input_shape=None):
        self.shape = shape
        self.extractor = extractor
        self.input_shape = input_shape if input_shape is not None else shape


class DatasetFeaturesSet(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, encoder, config):
        self.encoder = encoder
        self.config = config

    OPENSMILE_ComParE_2016 = TensorEncoder(), FeaturesSetConfig(shape=(30, 6373), extractor=None)
    AUDIO = TensorEncoder(), FeaturesSetConfig(shape=(30, 1, 48000), extractor=AudioFeatureExtractor.L3)
    VIDEO_FACE = TensorEncoder(), FeaturesSetConfig(shape=(480, 224, 224, 3), extractor=VideoFeatureExtractor.VGG_FACE)
    VIDEO_SCENE = TensorEncoder(), FeaturesSetConfig(shape=(480, 224, 224, 3), extractor=VideoFeatureExtractor.VGG)
    PULSE = TensorEncoder(), FeaturesSetConfig(shape=(741, 2, 1), extractor=VideoFeatureExtractor.PULSE)
    CLASS = TensorEncoder(), FeaturesSetConfig(shape=1, extractor=None)
    VIDEO_SCENE_R2PLUS1_FEATURES = TensorEncoder(), FeaturesSetConfig(shape=(113, 512), extractor=None)
