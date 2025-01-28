# general training pipeline for both task and data trained models
# explicitly integrated with ray here

import os
import torch
import lightning as pl
from torch.utils.data import DataLoader
from synthetic_datasets.tasks.CDM import CDM
from models.modules.rnn_module import frRNN
from synthetic_datasets.datamodules.task_datamodule import TaskDataModule
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

# INCREASE THE RESPONSE PERIOD

# create the task
# TODO: might need task_config
task = CDM()

# create the datamodule
# NOTE: write as 'param': tune.choice([]) (or tune.OTHER) for hyperparam tuning
data_config = {
    "task": task,  # this has to follow AbstractClass
    "data_dir": "./",
    "n_trials": 1000,
    "batch_size": 64,
    "num_workers": 4,
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "init_states": None
}

# can add more kwargs here
DataModule = TaskDataModule(data_config)  
DataModule.setup()

# get the dataloaders
train = DataModule.train_dataloader()   
val = DataModule.val_dataloader()
input_size, output_size = DataModule.data_shape()

# create the model
# NOTE: write as 'param': tune.choice([]) (or tune.OTHER) for hyperparam tuning
model_config = {
    "input_size": input_size,
    "hidden_size": None,
    "output_size": output_size,
    "noise_std": None,
    "alpha": 0.2,
    "rho": 1,
    "train_wi": False,
    "train_wo": False,
    "train_wrec": True,
    "train_h0": False,
    "train_si": True,
    "train_so": True,
    "wi_init": None,
    "wo_init": None,
    "wrec_init": None,
    "si_init": None,
    "so_init": None,
    "b_init": None,
    "add_biases": False,
    "non_linearity": torch.tanh,
    "output_non_linearity": torch.tanh,
    "lr": 1e-3,
    "weight_decay": 0.0
}

model = frRNN(model_config) 

# send to train with Ray
num_epochs = 100
grace_period = 1
reduction_factor = 2
scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

# training function
def train_func(model):
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

# # save final model
# mpath = os.path.join(data.data_dir, "model.pt")  # TODO: need specific name
# trainer.save_checkpoint("model.pt")

#
