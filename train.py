# general training pipeline for both task and data trained models

import logging
import os
import torch
from lightning import LightningModule
from lightning import Trainer
from lightning import seed_everything
from torch.utils.data import DataLoader
from synthetic_datasets.tasks.CDM import CDM
from models.modules.rnn_module import frRNN
from synthetic_datasets.datamodules.task_datamodule import TaskDataModule

# INCREASE THE RESPONSE PERIOD

# create the task
# TODO: might need task_config
task = CDM()

# create the datamodule
data_config = {
    "task": task,  # ote tis has to follow AbstractClass
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

# # create the trainer
# trainer = Trainer()

# # Ray - hyperparameter tuning
# # train
# trainer.fit(model, data)

# # save final model
# mpath = os.path.join(data.data_dir, "model.pt")  # TODO: need specific name
# trainer.save_checkpoint("model.pt")

#
