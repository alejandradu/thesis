from models.modules.rnn_module import frRNN
import torch

model_config = {
    "input_size": 4,
    "hidden_size": 10,
    "output_size": 1,
    "noise_std": 0.0,  # TODO: check what this noise is
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

full = frRNN(model_config)

# test one forward
inputs = torch.randn(20, 2500, 4)
outputs = full(inputs)
outputs2 = full(inputs)

# not detecting same bug as in train.py
