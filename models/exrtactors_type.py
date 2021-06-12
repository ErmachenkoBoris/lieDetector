from enum import Enum


class FeatureExtractorConfig:
    def __init__(self, output_shape):
        self.output_shape = output_shape


class AudioFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config):
        self.config = config

    L3 = FeatureExtractorConfig(output_shape=(10, 32, 24, 512))


class VideoFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config):
        self.config = config

    VGG = FeatureExtractorConfig(output_shape=())
    VGG_FACE = FeatureExtractorConfig(output_shape=())
    PULSE = FeatureExtractorConfig(output_shape=(741, 2))