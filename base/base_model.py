import logging
import os

import tensorflow as tf
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """ Базовый класс, описывающий модель машинного обучения """

    def __init__(self, cp_dir: str, cp_name: str, save_freq: int, model):
        self._cp_dir = cp_dir
        self._cp_name = cp_name
        self._save_freq = save_freq
        self.model = model
        self.epoch = 0

    def load(self):
        latest = tf.train.latest_checkpoint(self._cp_dir)
        if latest:
            logging.debug("Load checkpoint from: {}".format(latest))
            print(latest)
            checkpoint = self.model.load_weights(latest)
            self.epoch = self._get_epoch_from_name(latest)
            return checkpoint, self.epoch
        else:
            logging.debug("There is no checkpoint :(")
            return None, 0

    @staticmethod
    def _get_epoch_from_name(checkpoint_name):
        epoch_str = checkpoint_name[-9:-5]
        return int(epoch_str)

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def get_model_callbacks(self):
        return [self._get_save_model_callback()]

    def _get_save_model_callback(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self._cp_dir, self._cp_name),
            save_weights_only=True,
            cp_dir=self._cp_dir)

    @staticmethod
    def _get_early_stopping_callback():
        return tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')
