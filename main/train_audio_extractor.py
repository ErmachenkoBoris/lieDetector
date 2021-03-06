import os
import sys

from dataset.manager.data_manager import DataManager
from dataset.preprocessor.audio_modality_preprocessor import AudioModalityPreprocessor, AudioFeatureExtractor
from models.audio.audio_extractor_model import FineTuneModel
from trainers.audio_extractor_trainer import AudioExtractorTrainer

sys.path.insert(0, './')

import configs.deception_default_config as config


def main():
    processor = AudioModalityPreprocessor(extractor=AudioFeatureExtractor.L3)

    data_manager = DataManager(dataset_processor=processor,
                               dataset_size=config.DATASET_SIZE,
                               tf_record_path=os.path.join(config.DATASET_TF_RECORDS_PATH, config.NAME),
                               batch_size=config.BATCH_SIZE)

    model = FineTuneModel(
        extractor=AudioFeatureExtractor.L3,
        pretrained_model_path='../models/pretrained/openl3_audio_mel256_music.h5',
        cp_dir=config.CHECKPOINT_DIR,
        cp_name=config.CHECKPOINT_NAME,
        learning_rate=config.LEARNING_RATE,
        iter_per_epoch=config.NUM_ITER_PER_EPOCH
    )

    model.load()

    trainer = AudioExtractorTrainer(
        model=model,
        data=data_manager,
        board_path=config.TENSORBOARD_DIR,
        log_freq=config.LOG_AND_SAVE_FREQ_BATCH,
        num_epochs=config.NUM_EPOCHS,
        num_iter_per_epoch=config.NUM_ITER_PER_EPOCH,
        validation_steps=config.VALIDATION_STEPS,
        lr=config.LEARNING_RATE,
        create_dirs_flag=True
    )

    metrics = trainer.train()
    print(metrics)


if __name__ == "__main__":
    main()
