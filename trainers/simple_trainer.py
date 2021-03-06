from base.base_train import BaseTrain
from dataset.manager.data_manager import DataManager
from utils.logger import Logger


class SimpleTrainer(BaseTrain):

    def __init__(self,
                 model,
                 data: DataManager,
                 board_path: str,
                 log_freq: int,
                 num_epochs: int,
                 dataset_processor,
                 num_iter_per_epoch=None,
                 validation_steps=None,
                 create_dirs_flag=False,
                 initial_epoch=0):
        self._initial_epoch = initial_epoch
        self._dataset_processor = dataset_processor
        self._board_path = board_path
        self._log_freq = log_freq
        self._num_epochs = num_epochs
        self._num_iter_per_epoch = num_iter_per_epoch
        self._validation_steps = validation_steps
        self._create_dirs_flag = create_dirs_flag
        super(SimpleTrainer, self).__init__(model, data)

    def train(self):
        training_dataset = self.data.build_training_dataset(self._dataset_processor)
        validation_dataset = self.data.build_validation_dataset(self._dataset_processor)

        train_model = self.model.get_train_model()

        callbacks = [*self.model.get_model_callbacks(),
                     self._get_terminate_on_nan_callback(),
                     *Logger.get_logger_callbacks(self._board_path, self._log_freq, self._create_dirs_flag)]

        if not self._num_iter_per_epoch and not self._validation_steps:
            history = train_model.fit(
                training_dataset,
                steps_per_epoch=self._num_iter_per_epoch,
                initial_epoch=self._initial_epoch,
                epochs=self._num_epochs,
                validation_data=validation_dataset,
                validation_steps=self._validation_steps,
                callbacks=callbacks)
        else:
            history = train_model.fit(
                training_dataset,
                initial_epoch=self._initial_epoch,
                epochs=self._num_epochs,
                validation_data=validation_dataset,
                callbacks=callbacks)

        return history.history
