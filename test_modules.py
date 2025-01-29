from models.modules.rnn_module import frRNN
from synthetic_datasets.datamodules.task_datamodule import TaskDataModule
from synthetic_datasets.tasks.CDM import CDM
import torch

# setup the task
TASK_CONFIG = {
    "seed": 0,
    "coherences": None,
    "n_trials": 20,
    "bin_size": 10,
    "noise": 0.0,
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

DATA_CONFIG = {
    "task": task,  # this has to follow AbstractClass
    "data_dir": "/Users/alejandraduran/Documents/THESIS/thesis/",
    "n_trials": 20,
    "batch_size": 64,
    "num_workers": 4,  # difference between this and the num_workers in scaling_config?
    "train_ratio": 0.8,
    "val_ratio": 0.2,
    "init_states": None,
    "init_states_name": 'none',
    "init_states_dimension": 10,  # HAVE TO MATCH THIS WITH HIDDEN SIZE
} 

# model_config = {
#     "input_size": 4,
#     "hidden_size": 10,
#     "output_size": 1,
#     "noise_std": 0.0,  # TODO: check what this noise is
#     "alpha": 0.2,
#     "rho": 1,
#     "train_wi": False,
#     "train_wo": False,
#     "train_wrec": True,
#     "train_h0": False,
#     "train_si": True,
#     "train_so": True,
#     "wi_init": None,
#     "wo_init": None,
#     "wrec_init": None,
#     "si_init": None,
#     "so_init": None,
#     "b_init": None,
#     "add_biases": False,
#     "non_linearity": torch.tanh,
#     "output_non_linearity": torch.tanh,
#     "lr": 1e-3,
#     "weight_decay": 0.0
# }

# full = frRNN(model_config)

# # test one forward
# inputs = torch.randn(20, 2500, 4)
# outputs = full(inputs, return_latents=True)
# outputs2 = full(inputs, return_latents=False)

# print the input dataset from the datamodule
data_module = TaskDataModule(DATA_CONFIG)
data_module.prepare_data()
data_module.setup()

