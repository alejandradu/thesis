import os
import h5py
import logging
import lightning as L
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from task_data.AbstractClass import SyntheticTask

# Configure logging
logger = logging.getLogger(__name__)

class TaskDataModule(L.LightningDataModule):
    """Organize data creation and saving/loading to train a 
    task-trained network for one tasks
    
    TODO: extend to multitask?
    
    Args:
        task: SyntheticTask task to generate/handle the dataset
        data_dir: str, directory for data saving, end WITH "/"
    """
    def __init__(self, 
                 task: SyntheticTask,
                 data_dir: str = "./", 
                 n_trials: int = 1000,   # = n_samples for others
                 batch_size: int = 64, 
                 num_workers: int = 4, 
                 **kwargs):
        
        super().__init__()
        self.task = task
        self.data_dir = data_dir
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    # don't assign self. things here but in setup
    def prepare_data(self):
        
        # only load if same dataset already exists
        filename = os.path.join(self.data_dir, f"dataset_{hash(frozenset(self.kwargs.items()))}.h5")
        
        if os.path.exists(filename):
            logger.info(f"Loading dataset from {filename}")
            with h5py.File(filename, 'r') as f:
                data = {key: f[key][()] for key in f.keys()}
        else:
            logger.info(f"Generating dataset with specifications: {self.kwargs}")
            data = self.generate_dataset(self.kwargs)
            os.makedirs(self.data_dir, exist_ok=True)
            with h5py.File(filename, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)
            logger.info(f"Dataset saved to {filename}")
            
        return data

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.kwargs['batch_size'])