# load the result grid and analyze different (practical) aspects of the
# training itself and the evolution of metrics

import sys 
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ray import tune
from scripts_to_run.train_cluster import train_loop

class Analyzer():
    def __init__(self, experiment_path, lightning_module):
        """Load the best model of a tuning session and analyze its latents
        
        Args:
            checkpoint_dir (str): path to the train session full ~/ray_results/TorchTrainer[...]
            lightning_module (LightningModule): the model that was optimized with this TorchTrainer
        """
        self.experiment_path = experiment_path
        self.lightning_module = lightning_module
        self.best_model = None
        self.best_model_metadata = None
        self.result_grid = None
        
    def load_result_grid(self):
        # load the result grid from the TorchTrainer experiment
        restored_tuner = tune.Tuner.restore(self.experiment_path, trainable=train_loop)
        self.result_grid = restored_tuner.result_grid()
    
    def load_best_model(self, metric=None, mode=None):
        """Load the best model from the best checkpoint in Result
            and return the best checkpoint's metadata
            metric (str): The key for checkpoints to order on
            mode (str): One of [“min”, “max”]
        """
        if self.result_grid == None:
            self.load_result_grid()
        best_result = self.result_grid.get_best_result()
        best_checkpoint = best_result.get_best_checkpoint(metric=metric, mode=mode)
        with best_checkpoint.as_directory() as dir:
            self.best_model = dir['model.pt']
        self.best_model_metadata = best_checkpoint.get_metadata()

    def model(self, input, initial_states=None):
        """Run inference, mimic the original torch interface.
        Return (readout, trajectories) for the given input and inits"""
        if self.model == None:
            self.load_model()
        # disable batch normalization, dropout, randomness
        self.model.eval()
        return self.model(input, return_latents=True, initial_states=initial_states)

    def plot_trajectories(self, trajs, pca=False, tsne=False):
        # trajs have shape (n_timesteps, n_trials, hidden_size = n_neurons)
        trials, timesteps, dimensions = trajs.shape
        colors = plt.cm.viridis(np.linspace(0, 1, trials))  # Generate different colors for each trial

        fig = plt.figure(figsize=(10, 6))

        if dimensions == 2:
            ax = fig.add_subplot(111)
            for trial in range(trials):
                ax.scatter(trajs[trial, :, 0], trajs[trial, :, 1], color=colors[trial])#, label=f'Trial {trial+1}')
            ax.set_xlabel('latent 1')
            ax.set_ylabel('latent 2')
        elif dimensions == 3:
            ax = fig.add_subplot(111, projection='3d')
            for trial in range(trials):
                ax.scatter(trajs[trial, :, 0], trajs[trial, :, 1], trajs[trial, :, 2], color=colors[trial])#, label=f'Trial {trial+1}')
            ax.set_xlabel('latent 1')
            ax.set_ylabel('latent 2')
            ax.set_zlabel('latent 3')
        else:
            raise ValueError("FOR NOW Dimensions must be 2 or 3 for plotting.")
        
        # TODO: implement pca/tsne

        return fig
                
    def plot_targets(self, inputs, targets, readout, phase_index=None):
        
        # TODO: generalize, this is labeled only for the CDM task
        
        inputs = inputs.squeeze().numpy() 
        targets = targets.squeeze().numpy()
        
        # plot the inputs and targets
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(inputs, label=['Channel 0', 'Channel 1', 'Context 0', 'Context 1'])
        ax[0].set_xlim(0, len(inputs))
        ax[0].legend()
        ax[1].plot(targets, color='k', label='targets')
        ax[1].plot(readout, color='r', label='readout')  # BUG: maybe this sould NOT be the readout??
        
        # plot vertical dotted lines at x values of phase_index
        if phase_index:
            for key, val in phase_index.items():
                ax[0].axvline(x=val, color='k', linestyle='--')
                ax[1].axvline(x=val, color='k', linestyle='--')
            
        # add labels
        ax[0].set_xlabel("Binned timesteps (bin_size = {})".format(self.bin_size))
        ax[0].set_ylabel("Input amplitude")
        ax[1].set_xlabel("Binned timesteps (bin_size = {})".format(self.bin_size))
        ax[1].set_ylabel("Target vs Predictions")
        
        # return the image
        return fig
