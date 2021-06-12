from kapre.time_frequency import Melspectrogram
from keras_vggface import VGGFace

from base.base_model import BaseModel
import tensorflow as tf

from models.exrtactors_type import AudioFeatureExtractor, VideoFeatureExtractor


class UnimodalModel(BaseModel):

    def __init__(self,
                 modality,
                 fc_units,
                 first_layer,
                 second_layer,
                 cp_dir: str,
                 cp_name: str,
                 dropout=0.1,
                 pretrained_model_path: str = "",
                 num_fine_tuned_layers: int = 2,
                 log_and_save_freq_batch: int = 100,
                 learning_rate: float = 0.001,
                 trained: bool = False):

        self._modality = modality

        if self._modality.config.extractor is not None:
            print("Init model with trained features extractor... (WARN: extractor must be non null)")

            if self._modality.config.extractor == AudioFeatureExtractor.L3:
                self._pretrained_feature_extractor = tf.keras.models.load_model(pretrained_model_path,
                                                                                custom_objects={
                                                                                    'Melspectrogram': Melspectrogram},
                                                                                compile=False)
                self._pretrained_feature_extractor.trainable = False
            elif self._modality.config.extractor == VideoFeatureExtractor.VGG:
                self._pretrained_feature_extractor = tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                                                                 input_shape=self._modality.config.input_shape[
                                                                                             1:])
                self._pretrained_feature_extractor.trainable = False

            elif self._modality.config.extractor == VideoFeatureExtractor.VGG_FACE:
                self._pretrained_feature_extractor = VGGFace(include_top=False, weights='vggface',
                                                             input_shape=self._modality.config.input_shape[1:],
                                                             pooling='avg')
                self._pretrained_feature_extractor.trainable = False

        self._num_fine_tuned_layers = num_fine_tuned_layers
        self._learning_rate = learning_rate
        self._dropout = dropout

        self._optimizer = tf.keras.optimizers.Adam

        self._input_shape = self._modality.config.input_shape
        self._lstm_units_first_layer = first_layer
        self._lstm_units_second_layer = second_layer
        self._activation = 'relu'
        self._fc_units = fc_units

        self.train_model, self.test_model = self._build_model()
        self.model = self.test_model if trained else self.train_model

        super(UnimodalModel, self).__init__(cp_dir=cp_dir,
                                            cp_name=cp_name,
                                            save_freq=log_and_save_freq_batch,
                                            model=self.model)

    def _build_model(self):
        input_tensor = tf.keras.layers.Input(shape=self._input_shape)
        if self._modality.config.extractor is not None and self._modality.config.extractor != VideoFeatureExtractor.PULSE:
            for layer in self._pretrained_feature_extractor.layers[-self._num_fine_tuned_layers:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
            # (None, 480, 224, 224, 3)
            x = tf.map_fn(lambda frame_data: self._pretrained_feature_extractor(frame_data), input_tensor)
        elif self._modality.config.extractor is not None and self._modality.config.extractor == VideoFeatureExtractor.PULSE:
            x = self.pulse_conv_net(input_tensor)
        else:
            x = input_tensor

        print(x.shape)
        x = self.apply_lstm(x)
        x = tf.keras.layers.Dense(units=self._fc_units, activation=self._activation)(x)
        output_tensor = tf.keras.layers.Dense(units=1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
        model.summary()
        return model, model

    def get_train_model(self):
        metrics = ['accuracy', 'recall', 'precision', 'f1']
        self.train_model.compile(
            optimizer=self._optimizer(learning_rate=self._learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )
        return self.train_model

    def get_test_model(self):
        return self.test_model

    def apply_lstm(self, x):
        x = tf.keras.layers.LSTM(units=self._lstm_units_first_layer, return_sequences=True, dropout=self._dropout)(x)
        x = tf.keras.layers.LSTM(units=self._lstm_units_second_layer)(x)
        return x

    def pulse_conv_net(self, input_tensor):
        x = tf.keras.layers.Conv2D(16,
                                   kernel_size=(self._input_shape[0] // 2, 2),
                                   strides=(2, 1),
                                   activation=self._activation)(input_tensor)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 1))(x)
        return x
