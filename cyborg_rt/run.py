"""
run.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""
import os
import json
from time import sleep
from collections import OrderedDict
from datetime import datetime

from pytorch_lightning import seed_everything

from cyborg_rt.model import CYBORGxSAL
from cyborg_rt.train import create_trainer
from cyborg_rt.utils import get_logger
from cyborg_rt.data import DeepFakesDataModule
from cyborg_rt.data import get_test_data_loaders

logger = get_logger(__name__)


def run(C):
    logger.info(f'Global config:\n{C}')

    logger.info('Create trainer.')
    # create the logger here actually
    trainer, checkpoint_callback = create_trainer(C)

    seed_everything(C.SEED, workers=True)

    logger.info('Create model.')
    model = CYBORGxSAL(C)

    logger.info('Create data module.')
    data = DeepFakesDataModule(C)

    if C.HPARAM_TUNE:
        logger.info('Tuning hyperparameters before training.')
        trainer.tune(model)

    logger.info('Fine-tune model.')
    trainer.fit(model, data)

    logger.info('Load weights of best model.')
    best_model_path = checkpoint_callback.best_model_path

    retries = 10
    sleep_secs = 3
    while retries:
        retries -= 1
        try:
            model = CYBORGxSAL.load_from_checkpoint(best_model_path, C=C)
        except FileNotFoundError:
            logger.warning(f'Could not find "{best_model_path}"! Sleeping for '
                           f'{sleep_secs} seconds - hopefully filesystem '
                           f'synchronizes by then...This warning is far more '
                           f'common when DDP strategy is used.')
            sleep(sleep_secs)
        else:
            break

    logger.info('Test the best model.')
    test_data_loaders, test_dataset_names = get_test_data_loaders(C)
    results = trainer.test(model, test_data_loaders)
    results_processed = OrderedDict()
    for idx, name in enumerate(test_dataset_names):
        print(f'DataLoader {idx}: {name}')
        results_processed[name] = results[idx]

    results_dir = os.path.join(C.LOG_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)

    now_str = datetime.now().isoformat(timespec='seconds').replace(':', '_')    
    results_path = os.path.join(results_dir, 'scores_' + str(C.SEED) + '_' + now_str + '.json')
    config_path = os.path.join(results_dir, 'config_' + str(C.SEED) + '_' + now_str + '.json')

    print(f'Writing config to {config_path}')
    with open(config_path, 'w') as f:
        json.dump(C.as_dict(), f, indent=4)

    print(f'Writing results to {results_path}')
    with open(results_path, 'w') as f:
        json.dump(results_processed, f, indent=4)
