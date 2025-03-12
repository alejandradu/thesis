# general training pipeline for both task and data trained models
# explicitly integrated with ray here

import ray
import torch
import lightning as pl
from synthetic_datasets.tasks.NBFF import NBFF
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
num_epochs = 100
grace_period = 1
reduction_factor = 2
num_workers = 8   # SET THE SAME AS CPU
num_samples = 6  # total hyperparam combs 
hidden_size = 128
# init_states_dimension = hidden_size

# special to node 
latent_size = 2   # to plot, & change hidden size
init_states_dimension = latent_size
######## 

# setup the task
TASK_CONFIG = {
    "seed": 0,
    "n": 2,
    "n_trials": 2000,    # check that this > batch_size below
    "bin_size": 1,   # this is bin for TIMESTEPS
    "noise": 0.0,   # this is noise for the task itself
    "n_timesteps": 1000,
    "switch_prob": 0.01,
}

# create task
task = NBFF(TASK_CONFIG)
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
    "init_states_dimension": init_states_dimension,
    "init_states_name":'nbff_NODE',  # NOTE: have to change for nodes bc dimension of init states changes
} 

# setup the model
# NOTE: write as 'param': tune.choice([]) (or tune.OTHER) for hyperparam tuning
MODEL_CONFIG = {
    "model_class": nODE,
    "lr": tune.choice([1e-4, 1e-3, 1e-5]),
    "weight_decay": tune.choice([0.0, 1e-3]),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    
    # fr/lrRNN
    "noise_std": 0.0,  # this is noise for the evolution of the hidden states
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
    
    # lrRNN
    "rank": 2,
    "m_init": None,
    "n_init": None,
    
    # node
    "latent_size": latent_size,
    "output_mapping": None,  # default is to Linear. Change if you want a nonlinearity
    "num_layers": tune.choice([3, 10]),
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

# # FIX TO MANUALLY CONTROL THE PREPARE_DATA
# data_module = TaskDataModule(DATA_CONFIG)
# data_module.prepare_data()

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
def get_ray_trainer():
    return TorchTrainer(
        train_loop,
        scaling_config=SCALING_CONFIG,
        run_config=RUN_CONFIG,
    )
    
    
# note that the task is stated out of the pipline
def tune_pipeline():
    
    ray_trainer = get_ray_trainer()
        
    tuner = tune.Tuner(
        ray_trainer,
        # here goes the model_config
        param_space={"train_loop_config": MODEL_CONFIG},  # = train_loop(model_config)
        tune_config=tune.TuneConfig(
            metric="ptl/val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=SCHEDULER,
        ),
    )
    return tuner.fit()


if __name__ == "__main__":

    # run the pipeline and store result object in memory
    result_grid = tune_pipeline()

