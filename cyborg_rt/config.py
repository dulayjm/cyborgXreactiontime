"""
config.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael

Usage:
>>> from cyborg_sal.config import C
>>> C.LEARNING_RATE
"""
__all__ = ['C']

import inspect
from pathlib import Path
from pprint import pformat


class _Config:
    """Global config that can be modified"""
    SEED = 0xFACE

    LEARNING_RATE = 0.005
    LEARNING_RATE_DECAY_STEP_SIZE = 12  # StepLR step_size (epochs)
    LEARNING_RATE_DECAY_GAMMA = 0.1  # StepLR gamma
    OPTIMIZER = 'SGD'
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-6
    EPOCHS = 50
    BATCH_SIZE = 20
    STOCHASTIC_WEIGHT_AVERAGING = False

    BINARY_OUTPUT = False

    BACKBONE = 'DenseNet121'

    LOSS = 'CE'

    CYBORG_LOSS_ALPHA = 0.5
    PSYCH_SCALING_CONSTANT = 0.9
    USE_RANDOM_REACTIONTIME = 'random'
    
    GPUS = 1

    DEBUG = False
    QUICK_TEST = False

    HPARAM_TUNE = False

    # LOG_DIR = str(Path('./logs/').absolute())
    LOG_DIR = '/scratch365/jdulay/cyborg_logs/'
    SHOW_PROGRESS_BAR = DEBUG

    USE_WANDB_LOGGER = False

    DATA_DIR_BASE = '/scratch365/aboyd3/DataSynFace/ALL_IMAGES/'
    DATA_DIR_TRAIN = 'train'
    # DATA_DIR_TRAIN_ANNOTATIONS = 'heatmaps'
    DATA_DIR_TRAIN_ANNOTATIONS = '/scratch365/aboyd3_new/DataSynFace/heatmap_ablation/original/aligned/correct_annotations/'
    TRAIN_SPLIT = 'original_data'
    DATA_DIR_VAL = 'val'
    DATA_DIR_TEST = 'test'
    TEST_DATASET_IDXS = 6
    DATA_DIR_REAL = '0_real'
    DATA_DIR_FAKE = '1_fake'
    REACTIONTIME_FILE = str(Path('/scratch365/jdulay/annotation_cyborg_time_data/annotations.csv').absolute())
    REACTIONTIME_BRIDGE_FILE = str(Path('/scratch365/jdulay/annotation_cyborg_time_data/images.csv').absolute())

    # You can safely ignore everything below if you're modifying default values
    __C = None

    def __new__(cls, *args, **kwargs):
        if cls.__C is None:
            cls.__C = super().__new__(cls, *args, **kwargs)
            return cls.__C
        # otherwise
        raise RuntimeError(f'The class {cls} can only be instantiated once!')

    def keys(self):
        return self.as_dict().keys()

    def values(self):
        return self.as_dict().values()

    def as_dict(self):
        return dict(attr for attr in
                    inspect.getmembers(
                        self, lambda attr: not inspect.isroutine(attr))
                    if not attr[0].startswith('_'))

    def __str__(self):
        return pformat(self.as_dict())

    def __repr__(self):
        return str(self)


C = _Config()
