#!/usr/bin/env python
"""
main.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""

import os

if __name__ == '__main__':
    import argparse
    from cyborg_rt.config import C

    parser = argparse.ArgumentParser(
        description='Entry script to CYBORG-RT experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # global config
    config_dict = C.as_dict()
    parser.add_argument(
        '--seed', help='The seed to pass into the config file',
        default=config_dict.pop('SEED'),
    )
    parser.add_argument(
        '-B', '--backbone', help='The name of the pre-trained backbone to use.',
        choices=['DenseNet121', 'ResNet50', 'Inception_v3', 'Xception',
                 'CNNDetection', 'Self-Attention'],
        default=config_dict.pop('BACKBONE'),
    )
    parser.add_argument(
        '-L', '--loss', help='The name of the loss to use.',
        choices=['BCE', 'CE', 'CYBORG', 'CYBORG+REACTIONTIME', 
                'REACTIONTIME', 'CYBORG+HARMONIZATION'],
        default=config_dict.pop('LOSS'),
    )
    parser.add_argument(
        '-T', '--train-split',
        help='The sub-directory name of the training data.',
        choices=['original_data', '6x_data', '7x_data'],
        default=config_dict.pop('TRAIN_SPLIT'),
    )
    parser.add_argument(
        '--debug', type=bool,
        help='whether to print verbosely',
        default=config_dict.pop('DEBUG'),
    )
    parser.add_argument(
        '--use-random-reactiontime',
        help='Whether to use random reaction times instead of measurements from human data.',
        choices=['random', 'normal'],
        default=config_dict.pop('USE_RANDOM_REACTIONTIME'),
    )
    parser.add_argument(
        '--use-wandb-logger',
        type=bool, 
        choices=[True,False],
        default=config_dict.pop('USE_WANDB_LOGGER'),
    )
    for param, default in config_dict.items():
        param_name = param.lower().replace('_', '-')
        type_ = type(default)
        if type_ is bool:
            param_name = (f'--{param_name}' if not default else
                          f'--no-{param_name}')
            kwargs = {'action': 'store_true', 'dest': param.lower()}
        else:
            param_name = f'--{param_name}'
            kwargs = {'type': type_, 'default': default}
        parser.add_argument(
            param_name, help=f'{param}', **kwargs
        )
    args = parser.parse_args()

    # now, update global config with user-supp
    # lied arguments
    for param in C.keys():
        value = getattr(args, param.lower())
        setattr(C, param, value)

    # delay larger imports until this point (to e.g. speed up displaying help)
    from cyborg_rt.run import run
    

    C.LOG_DIR = '/scratch365/jdulay/cyborg_logs/' \
        + C.BACKBONE \
        + '_' \
        + C.LOSS \
        + '_' \
        + 'QuartileNorm_AND_RTsampling' \
        + C.USE_RANDOM_REACTIONTIME \
        + '/'

    # if it doesn't exist, make it
    if not os.path.exists(C.LOG_DIR):
        os.mkdir(C.LOG_DIR)
    print("Logging results and checkpoints to: ", C.LOG_DIR)

    run(C)
