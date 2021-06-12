import sys

from configs.dataset.modality import DatasetFeaturesSet
from dataset.manager.data_manager import DataManager
from dataset.preprocessor.multimodal_dataset_features_processor import MultimodalDatasetFeaturesProcessor
from models.unimodal_model import UnimodalModel
from trainers.simple_trainer import SimpleTrainer

sys.path.insert(0, './')

import configs.local_config as config


def main():
    processor = MultimodalDatasetFeaturesProcessor(modalities_list=[DatasetFeaturesSet.AUDIO])

    data_manager = DataManager(
        tf_record_path=config.DATASET_TF_RECORDS_PATH,
        batch_size=1)

    model = UnimodalModel(
        modality=DatasetFeaturesSet.AUDIO,
        fc_units=64,
        first_layer=128,
        second_layer=64,
        learning_rate=0.001,
        pretrained_model_path=DatasetFeaturesSet.AUDIO.config.extractor.config.pretrained_path,
        cp_dir=config.CHECKPOINT_DIR,
        cp_name=config.CHECKPOINT_NAME
    )

    _, epoch = model.load()

    trainer = SimpleTrainer(
        dataset_processor=processor,
        model=model,
        data=data_manager,
        board_path=config.TENSORBOARD_DIR,
        log_freq=config.LOG_AND_SAVE_FREQ_BATCH,
        num_epochs=2,
        initial_epoch=epoch,
        create_dirs_flag=True
    )

    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()
