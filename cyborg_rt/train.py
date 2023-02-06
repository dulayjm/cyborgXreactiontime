"""
create_trainer.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from cyborg_rt.utils import get_logger

logger = get_logger(__name__)


def create_trainer(C):
    """

    """    
    if C.USE_WANDB_LOGGER:
        proj_name = C.BACKBONE \
            + '_' \
            + C.LOSS \
            + '_' \
            + str(C.PSYCH_SCALING_CONSTANT) \
            + C.USE_RANDOM_REACTIONTIME \
            + str(C.SEED)

        # let's create a WandBLogger
        wandb_logger = WandbLogger(
            name=proj_name,
            project="v6CYBORG",  
            save_dir=C.LOG_DIR,
            log_model=True
        )
        wandb_logger.experiment.config.update(C.as_dict())
    else:
        # just set to the expected default falue for the trainer
        wandb_logger = True

    # TODO: do custom names for checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val_auc',
        # monitor='val_accuracy',
        mode='max',
    )
    callbacks = [checkpoint_callback]
    if C.STOCHASTIC_WEIGHT_AVERAGING:
        callbacks.append(StochasticWeightAveraging())

    logger.info(f'Logging checkpoints to "{C.LOG_DIR}"')
    kwargs = {}
    if C.HPARAM_TUNE:
        # Currently, this feature supports two modes ‘power’ scaling and
        # ‘binsearch’ scaling. In ‘power’ scaling, starting from a batch size
        # of 1 keeps doubling the batch size until an out-of-memory (OOM) error
        # is encountered. Setting the argument to ‘binsearch’ will initially
        # also try doubling the batch size until it encounters an OOM, after
        # which it will do a binary search that will finetune the batch size.
        # Additionally, it should be noted that the batch size scaler cannot
        # search for batch sizes larger than the size of the training dataset.
        kwargs['auto_scale_batch_size'] = 'binsearch'
        
    trainer = Trainer(
        default_root_dir=C.LOG_DIR,
        logger=wandb_logger,
        callbacks=callbacks,
        devices=C.GPUS,
        accelerator='gpu' if C.GPUS > 0 else 'cpu',
        # DistributedDataParallel if using more than 1 GPU
        strategy=None if C.GPUS < 2 else 'ddp',
        max_epochs=C.EPOCHS,
        num_sanity_val_steps=2 if C.DEBUG else 0,
        # we don't want to generate gigantic o.* files
        enable_progress_bar=False,
        # for debugging only 
        # limit_train_batches=0
    )
    return trainer, checkpoint_callback
