# general training pipeline for both task and data trained models
# explicitly integrated with ray here

import ray
import torch
import lightning as pl
from torch.utils.data import DataLoader
from synthetic_datasets.tasks.CDM import CDM
from models.modules.rnn_module import *
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

######## CONFIGS
num_epochs = 50
grace_period = 1
reduction_factor = 2
num_workers = 8   # SET THE SAME AS CPU
num_samples = 8  # total hyperparam combs 
######## 

# setup the task
TASK_CONFIG = {
    "seed": 0,
    "coherences": None,
    "n_trials": 2000,    # check that this > batch_size below
    "bin_size": 10,   # this is bin for TIMESTEPS
    "noise": 0.0,   # this is noise for the task itself
    "n_timesteps": 250+800+2000+250+200,
    "fix": 250,
    "ctx": 800,
    "stim": 2000,
    "mem": 250,
    "res": 200,
    "random_trials": False,
    "ctx_choice": None,
    "coh_choice0": None,
    "coh_choice1": None,
    "coh_scale": 1e-1,
    "ctx_scale": 1e-1
}

# create task
task = CDM(TASK_CONFIG)
input_size = task.input_size
output_size = task.output_size
 
# setup the datamodule
# NOTE: write as 'param': tune.choice([]) (or tune.OTHER) for hyperparam tuning
# can merge with model_config as dict to optimize over it - not expecting to need this
DATA_CONFIG = {
    "task": task,  # this has to follow AbstractClass
    "data_dir": "/scratch/gpfs/ad2002/task_training/task_data/",
    "batch_size": 64,   # COMPARE WITH N TRIALS SET FOR TASK   # NOTE: make this more logical later
    "num_workers": num_workers,  # difference between this and the num_workers in scaling_config?
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "init_states": None,
    "init_states_dimension": 10,  # HAVE TO MATCH THIS WITH HIDDEN SIZE
    "init_states_name":'TEST',
} 

# setup the model
# NOTE: write as 'param': tune.choice([]) (or tune.OTHER) for hyperparam tuning
MODEL_CONFIG = {
    "model_class": frRNN,
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

N_TIMESTEPS = TASK_CONFIG["n_timesteps"]
BIN_SIZE = TASK_CONFIG["bin_size"]

# optimize trials (stop if params are likely to be bad)
SCHEDULER = ASHAScheduler(time_attr='training_iteration', max_t=num_epochs,
                              grace_period=grace_period, reduction_factor=reduction_factor)

# HERE for distributed training
SCALING_CONFIG = ScalingConfig(
    num_workers=num_workers, use_gpu=False, resources_per_worker={"CPU": 1, "GPU": 0.0}   # divide by worker
)

RUN_CONFIG = RunConfig(
    storage_path="/scratch/gpfs/ad2002/task_training/ray_results/",
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="ptl/val_accuracy",
        checkpoint_score_order="max",
    ),
)

# training function to execute on each worker
def train_loop(model_config):

    # create the model
    model = GeneralModel(model_config)
    # create data: encapsulate all train, val, test splits
    data_module = TaskDataModule(DATA_CONFIG) 
    
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


# Define a TorchTrainer without hyper-parameters for Tuner
# this is passed to recover the TorchTrainer results
def get_ray_trainer(train_loop, scaling_config, run_config):
    return TorchTrainer(
        train_loop,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    
    
# note that the task is stated out of the pipline
def tune_pipeline():
    
    ray_trainer = get_ray_trainer(train_loop, SCALING_CONFIG, RUN_CONFIG)
        
    tuner = tune.Tuner(
        ray_trainer,
        # here goes the model_config
        param_space={"train_loop_config": MODEL_CONFIG},  # = train_loop(model_config)
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=SCHEDULER,
        ),
    )
    return tuner.fit()


if __name__ == "__main__":

    # run the pipeline and store result object in memory
    result_grid = tune_pipeline(num_samples=num_samples)

