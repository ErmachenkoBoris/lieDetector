EXP_NAME = "unimodal_deception"

DATASET_TF_RECORDS_PATH = "D:/2021/hse/lie-detector/deception_dataset/tf-records"

CHECKPOINT_DIR = "../experiments/" + EXP_NAME + "/checkpoint/"
CHECKPOINT_NAME = "cp-{epoch:04d}.ckpt"

TENSORBOARD_DIR = "../experiments/" + EXP_NAME + "/tb"
TENSORBOARD_NAME = 'epoch-{}'

PRETRAINED_MODELS = "D:/2021/hse/lie-detector/lieDetector/models/pretrained"
LOG_AND_SAVE_FREQ_BATCH = 10
