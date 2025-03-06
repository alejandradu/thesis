import torch
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from AbstractClass import SyntheticTask

# NOTE: generate docstrings with codeium, the period args are fractions of the trial


class NBFF(SyntheticTask):
    """Represent an n-bit flip flop task: the network has to remember
    the most recent bit combination from the channels"""
        
    def __init__(self, task_config):
        
        super().__init__()
        
        # save config dictionary to hash name of dataset
        self.task_config = task_config
        
        self.seed = task_config["seed"]
        self.n = task_config["n"]
        self.switch_prob = task_config["switch_prob"]

        self.input_size = self.n   # n flipping channels
        self.output_size = self.n  # output the bit combination
        
        self.n_trials = task_config["n_trials"]
        self.bin_size = task_config["bin_size"]
        self.noise = task_config["noise"]
        self.n_timesteps = task_config["n_timesteps"]
    
        
    def generate_dataset(self, to_plot=False):
        
        # only generate one to plot
        if to_plot:
            n_trials = 1
        else:
            n_trials = self.n_trials
            
        # random inputs 
        inputRand = np.random.random(size=(n_trials, self.n_timesteps, self.n))
        inputs = torch.zeros((n_trials, self.n_timesteps, self.n))
        inputs[
            inputRand > (1 - self.switch_prob)
        ] = 1  # 2% chance of flipping up or down
        inputs[inputRand < (self.switch_prob)] = -1
        
        # Set the first 3 inputs to 0 to make sure no inputs come in immediately
        inputs[:, 0:3, :] = 0
        
        # Generate the desired outputs (targets) given the inputs
        targets = torch.zeros_like(inputs)
        for t in range(n_trials):
            for channel in range(self.n):
                last_non_zero = 0  # Initialize the last non-zero input
                for i in range(3, self.n_timesteps):
                    current_in = inputs[t, i, channel]
                    if current_in != 0:
                        last_non_zero = current_in
                    if last_non_zero == 1:
                        targets[t, i, channel] = 1
                    elif last_non_zero == -1:
                        targets[t, i, channel] = 0

        # Add noise to the inputs for the trial - return the noisy version
        inputs = inputs + np.random.normal(loc=0.0, scale=self.noise, size=inputs.shape)
        return inputs, targets
    
    
    def plot_trial(self):
        """"Generate randomly one trial and plot it."""
        
        inputs, targets = self.generate_dataset(to_plot=True)
        inputs = inputs.squeeze().numpy() 
        targets = targets.squeeze().numpy()
        
        # plot the inputs and targets
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(inputs)
        ax[0].set_title("Inputs")
        ax[0].set_xlim(0, len(inputs))
        ax[0].legend()
        ax[1].plot(targets)
        ax[1].set_title("Targets")
            
        # add labels
        ax[0].set_xlabel("Binned timesteps (bin_size = {})".format(self.bin_size))
        ax[0].set_ylabel("Amplitude")
        ax[1].set_xlabel("Binned timesteps (bin_size = {})".format(self.bin_size))
        ax[1].set_ylabel("Target")
        
        # # return the image
        # return fig
        
        
