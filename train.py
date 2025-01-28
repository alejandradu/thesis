# general training pipeline for both task and data trained models
# explicitly integrated with ray here

import os
import torch
import lightning as pl
from torch.utils.data import DataLoader
from synthetic_datasets.tasks.CDM import CDM
from models.modules.rnn_module import frRNN
from synthetic_datasets.datamodules.task_datamodule import TaskDataModule
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray import train, tune
from ray.train.torch import TorchTrainer
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
    "n_trials": 20,
    "batch_size": 64,
    "num_workers": 4,  # difference between this and the num_workers in scaling_config?
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "init_states": None
}

# can add more kwargs here
DataModule = TaskDataModule(data_config)  
DataModule.setup()

# get the dataloaders
train_datamodule = DataModule.train_dataloader()   
val_datamodule = DataModule.val_dataloader()
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

######## train with Ray
num_epochs = 100
grace_period = 1
reduction_factor = 2
num_workers = 4   # SET THE SAME AS CPU
num_samples = 1  # this matters for other than tune.choice
######## 

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
    trainer.fit(model, train_dataloader=train_datamodule, val_dataloader=val_datamodule)
    
# optimize trials (stop if params are likely to be bad)
scheduler = ASHAScheduler(max_t=num_epochs, grace_period=grace_period, reduction_factor=reduction_factor)

# HERE for distributed training
scaling_config = ScalingConfig(
    num_workers=num_workers, use_gpu=False, resources_per_worker={"CPU": 1, "GPU": 1}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_accuracy",
        checkpoint_score_order="max",
    ),
)

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_mnist_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        # here goes the model_config
        param_space={"train_loop_config": model_config},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


results = tune_mnist_asha(num_samples=num_samples)

# # save final model
# mpath = os.path.join(data.data_dir, "model.pt")  # TODO: need specific name
# trainer.save_checkpoint("model.pt")

#
