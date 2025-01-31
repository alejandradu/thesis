# general training pipeline for both task and data trained models
# explicitly integrated with ray here

import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# setup the task
TASK_CONFIG = {
    "seed": 0,
    "coherences": None,
    "n_trials": 2000,    # check that this > batch_size below
    "bin_size": 20,   # this is bin for TIMESTEPS
    "noise": 0.0,   # this is noise for the task itself
    "n_timesteps": 250+800+2000+250+50,
    "fix": 250,
    "ctx": 800,
    "stim": 2000,
    "mem": 250,
    "res": 50,
    "random_trials": False,
    "ctx_choice": None,
    "coh_choice0": None,
    "coh_choice1": None,
    "coh_scale": 1e-1,
    "ctx_scale": 1e-1
}

# create task
task = CDM(TASK_CONFIG)
# task.plot_trial()

input_size = task.input_size
output_size = task.output_size

# setup the datamodule
# NOTE: write as 'param': tune.choice([]) (or tune.OTHER) for hyperparam tuning
# can merge with model_config as dict to optimize over it - not expecting to need this
DATA_CONFIG = {
    "task": task,  # this has to follow AbstractClass
    "data_dir": "/scratch/gpfs/ad2002/task_training/task_data/",
    "batch_size": 64,   # COMPARE WITH N TRIALS SET FOR TASK   # NOTE: make this more logical later
    "num_workers": 4,  # difference between this and the num_workers in scaling_config?
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "init_states": None,
    "init_states_dimension": 10,  # HAVE TO MATCH THIS WITH HIDDEN SIZE
    "init_states_name":'none',
} 

# setup the model
# NOTE: write as 'param': tune.choice([]) (or tune.OTHER) for hyperparam tuning
MODEL_CONFIG = {
    "input_size": input_size,
    "hidden_size": 10,
    "output_size": output_size,
    "noise_std": tune.choice([0.0, 0.1]),  # this is noise for the evolution of the hidden states
    "alpha": 1,     # this should be t/TAU?
    "rho": 1,       # this is for matrix initialization distributions
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
    "lr": tune.choice([1e-4, 1e-3]),
    "weight_decay": tune.choice([0.0, 1e-3]),
}

######## train with Ray
num_epochs = 50
grace_period = 1
reduction_factor = 2
num_workers = 4   # SET THE SAME AS CPU
num_samples = 1  # this matters for other than tune.choice
######## 

# training function
def train_loop(model_config):
    
    # create the model
    model = frRNN(model_config) 
    # create data: encapsulate all train, val, test splits
    data_module = TaskDataModule(DATA_CONFIG) 
    
    data_module.prepare_data()
    data_module.setup()
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=data_module)
    
# optimize trials (stop if params are likely to be bad)
scheduler = ASHAScheduler(max_t=num_epochs, grace_period=grace_period, reduction_factor=reduction_factor)

# HERE for distributed training
scaling_config = ScalingConfig(
    num_workers=num_workers, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 0.25}   # divide by worker
)

run_config = RunConfig(
    storage_path="/scratch/gpfs/ad2002/task_training/ray_results/",
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_accuracy",
        checkpoint_score_order="max",
    ),
)

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_loop,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_mnist_asha(num_samples=1):
    scheduler = ASHAScheduler(time_attr='training_iteration', max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        # here goes the model_config
        param_space={"train_loop_config": MODEL_CONFIG},  # = train_loop(model_config)
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()


results = tune_mnist_asha(num_samples=num_samples)

