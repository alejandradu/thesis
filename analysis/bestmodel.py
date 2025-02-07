# load the last best checkpoint and class Analysis:

class BestModel:
    def __init__(self, checkpoint_dir):
        """Load the best model of a tuning session and analyze its latents
        
        Args:
            checkpoint_dir (str): path to the TorchTrainer best checkpoint
        """
        self.checkpoint_dir = checkpoint_dir
        self.model = None

    def load_model(self):
        # Load the model with optimized parameters from the checkpoint
        # Use torch.load to load the checkpoint and extract the model
        pass

    def run_inference(self, input_dataset):
        # Run inference on the model with the input dataset
        # Use the model to predict outputs for the input dataset
        pass

    def plot_trajectories(self, predictions, time_steps):
        # Plot the trajectories in time
        # Use matplotlib to create the plot
        pass

    def analyze(self, input_dataset, time_steps):
        # High-level method to perform the entire analysis
        self.load_model()
        predictions = self.run_inference(input_dataset)
        self.plot_trajectories(predictions, time_steps)