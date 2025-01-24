# general training pipeline for both task and data trained models

import logging
import os
import torch
from lightning import LightningModule
from lightning import Trainer
from lightning import seed_everything
from torch.utils.data import DataLoader

# INCREASE THE RESPONSE PERIOD

# create the task
from synthetic_datasets.tasks.CDM import CDM
task = CDM()

# create the datamodule
from synthetic_datasets.datamodules.task_datamodule import TaskDataModule
data = TaskDataModule(task)  # THE ARGS??
# task: SyntheticTask,
#                  data_dir: str = "./", 
#                  n_trials: int = 1000,   # = n_samples for others
#                  batch_size: int = 64, 
#                  num_workers: int = 4, 
#                  train_ratio: float = 0.8,
#                  val_ratio: float = 0.2,
#                  init_states: torch.Tensor = None,
#                  **kwargs

# create the model
input_size = 10
hidden_size = 100
output_size = 1
noise_std = 0.05

input_size, output_size = data.data_shape()
print(input_size, output_size)

from models.modules.rnn_module import tradRNN
model = tradRNN(input_size, 
                hidden_size, 
                output_size, 
                noise_std, alpha=0.2, rho=1,
                 train_wi=False, train_wo=False, train_wrec=True, train_h0=False, train_si=True, train_so=True,
                 wi_init=None, wo_init=None, wrec_init=None, si_init=None, so_init=None, b_init=None,
                 add_biases=False, non_linearity=torch.tanh, output_non_linearity=torch.tanh, rank=128) 

# # create the trainer
# trainer = Trainer()

# # Ray - hyperparameter tuning
# # train
# trainer.fit(model, data)

# # save final model
# mpath = os.path.join(data.data_dir, "model.pt")  # TODO: need specific name
# trainer.save_checkpoint("model.pt")

#
