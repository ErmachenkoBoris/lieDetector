from enum import Enum

from configs.colab_config import PRETRAINED_MODELS


class FeatureExtractorConfig:
    def __init__(self, pretrained_path=""):
        self.pretrained_path = pretrained_path


class AudioFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config):
        self.config = config

    L3 = FeatureExtractorConfig(pretrained_path= PRETRAINED_MODELS + "/openl3_audio_mel256_env.h5")


class VideoFeatureExtractor(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, config):
        self.config = config

    VGG = FeatureExtractorConfig()
    VGG_FACE = FeatureExtractorConfig()
    PULSE = FeatureExtractorConfig()
