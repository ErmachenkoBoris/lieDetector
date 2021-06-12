EXP_NAME = "unimodal_deception"

DATASET_TF_RECORDS_PATH = "/content/drive/MyDrive/deception_dataset/tf-records/deception_dataset"

DATASET_PATH = "/content/drive/MyDrive/deception_dataset"

CHECKPOINT_DIR = DATASET_PATH + "/experiments/" + EXP_NAME + "/checkpoint/"
CHECKPOINT_NAME = "cp-{epoch:04d}.ckpt"

TENSORBOARD_DIR = DATASET_PATH + "/experiments/" + EXP_NAME + "/tb"
TENSORBOARD_NAME = 'epoch-{}'

PRETRAINED_MODELS = "/content/drive/MyDrive/pretrained"
LOG_AND_SAVE_FREQ_BATCH = 10
