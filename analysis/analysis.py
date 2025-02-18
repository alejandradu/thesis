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

class TuneResult:
    def __init__(self, experiment_path, lightning_module):
        """Load the best model of a tuning session and analyze its latents
        
        Args:
            checkpoint_dir (str): path to the train session full ~/ray_results/TorchTrainer[...]
            lightning_module (LightningModule): the model that was optimized with this TorchTrainer
        """
        self.experiment_path = experiment_path
        self.lightning_module = lightning_module
        self.model = None
        
    def load_result_grid(self):
        # load the result grid from the TorchTrainer experiment
        restored_tuner = tune.Tuner.restore(self.experiment_path, trainable=train_loop)
        return restored_tuner.result_grid()

    def load_model(self, checkpoint_path=None):
        # provide the checkpoint_path to immediately load a specific model
        if not checkpoint_path:
            # TODO: get all the checkpoints and iterate over folder
            print('Retry and provide the checkpoint_path for now')
            return
        self.model = self.lightning_module.load_from_checkpoint(checkpoint_path)
            
    def run_inference(self, input, targets, plot_trajs=False, plot_targets=False):
        if self.model == None:
            self.load_model()
        # disable batch normalization, dropout, randomness
        self.model.eval()
        readout, trajs = self.model(input, return_latents=True)
        if plot_trajs:
            self.plot_trajectories(trajs)
        return readout

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

        plt.title('Data Trajectories')
        plt.legend()
        plt.show()
                
    def plot_targets(self, readout, targets):
