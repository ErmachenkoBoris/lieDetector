import sys

from configs.dataset.modality import DatasetFeaturesSet
from dataset.manager.data_manager import DataManager
from dataset.preprocessor.multimodal_dataset_features_processor import MultimodalDatasetFeaturesProcessor
from models.multimodal_model import MultimodalModel
from models.unimodal_model import UnimodalModel
from trainers.simple_trainer import SimpleTrainer

sys.path.insert(0, './')

import configs.local_config as config


def main():
    modalities = [
        DatasetFeaturesSet.VIDEO_SCENE_R2PLUS1_FEATURES,
        DatasetFeaturesSet.VIDEO_FACE,
        DatasetFeaturesSet.AUDIO,
        DatasetFeaturesSet.PULSE,
        DatasetFeaturesSet.VIDEO_SCENE,
    ]

    fusion_types = ['sum', 'concatenation', 'fbp']
    for i in range((len(modalities))):
        for fusion in fusion_types:
            processor = MultimodalDatasetFeaturesProcessor(modalities_list=modalities[i:])

            data_manager = DataManager(
                tf_record_path=config.DATASET_TF_RECORDS_PATH,
                batch_size=1)

            model = MultimodalModel(
                fusion_type=fusion,
                modalities_list=modalities[i:],
                fc_units=64,
                first_layer=128,
                second_layer=64,
                learning_rate=0.0001,
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
