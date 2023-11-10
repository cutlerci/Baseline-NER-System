# pytorch lightning with wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint, LearningRateMonitor, \
    GradientAccumulationScheduler
import numpy as np
import torch
# if __name__ == '__main__':
#     torch.multiprocessing.set_start_method("spawn")
from pytorch_lightning.loggers import WandbLogger
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from older_gpu_continual_model import *
from continual_data_module import *

import random
import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"

torch.set_float32_matmul_precision('medium')
# torch.multiprocessing.set_sharing_strategy('file_system')

# set the seed
seed = 34
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if __name__ == '__main__':
    # print the result of pwd command
    print("Current working directory: {0}".format(os.getcwd()))

    # schedule = {0: 1, 4: 4, 8: 8}
    # schedule = {0: 2, 8: 10, 80: 40, 120: 60}
    schedule = {0: 1, }
    # model = WhisperEncoderPL(accumulation_schedule=schedule)

    #TODO use a loop to get this value
    CL_STEP = 0

    model = EncoderNERModel(accumulation_schedule=schedule)
    chkpt = ModelCheckpoint(monitor='valid_loss', mode='min', save_top_k=2, dirpath='./checkpoints/',
                            save_last=True, every_n_epochs=1, auto_insert_metric_name=True,
                            filename='{epoch}-{step}-{valid_loss:.2f}-{train_loss:.2f}')

    ChEMU = ChEMUDataModule("ChEMU2023_FOR_CLASS/ChEMU2023_FOR_CLASS", model.batch_size, True, False, CL_STEP)

    accumulator = GradientAccumulationScheduler(scheduling=schedule)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(project='ner_bert', log_model=True)
    # wandb_logger.watch(model, log_freq = 100, log_graph=False)

    trainer = pl.Trainer(accelerator="gpu", devices = 1, precision="16-mixed", enable_checkpointing=True,
                         gradient_clip_val=1, max_epochs=1000, check_val_every_n_epoch=1, num_sanity_val_steps=2, default_root_dir='./checkpoints/',
                         log_every_n_steps=1, logger=wandb_logger, callbacks=[lr_monitor, accumulator, chkpt,])

    trainer.fit(model, datamodule=ChEMU)
    trainer.validate(model, datamodule=ChEMU)
    # trainer.predict(model, datamodule=ChEMU)


