# general training pipeline for both task and data trained models

import logging
import os
from lightning import LightningModule
from lightning import Trainer
from lightning import seed_everything
from torch.utils.data import DataLoader

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
from base_models.rnn import FullRankRNN
model = FullRankRNN()  # THE ARGS??

# create the trainer
trainer = Trainer()

# train
trainer.fit(model, data)

# save final model
mpath = os.path.join(data.data_dir, "model.pt")  # TODO: need specific name
trainer.save_checkpoint("model.pt")
