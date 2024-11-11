import os
import h5py
import logging
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
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
        kwargs: timing parameters for the trials
    """
    def __init__(self, 
                 task: SyntheticTask,
                 data_dir: str = "./", 
                 n_trials: int = 1000,   # = n_samples for others
                 batch_size: int = 64, 
                 num_workers: int = 4, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2,
                 init_states: torch.Tensor = None,
                 **kwargs):
        
        super().__init__()
        self.task = task
        self.data_dir = data_dir
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.kwargs = kwargs
        self.init_states = init_states
        self.dpath = None

    # this will runk once: put complicated task generation here 
    def prepare_data(self):
        
        key = hash(frozenset(self.kwargs.items()))
        self.dpath = os.path.join(self.data_dir, f"task_{self.train_ratio}_{self.val_ration}_{key}.h5")
        
        # if data with same timing and splits already exists pass
        if os.path.exists(self.dpath):
            return
        else:
            logger.info(f"Generating dataset with settings and splits: {self.kwargs}, 
                        \n{self.train_ratio}, {self.val_ration}, {self.test_ratio}")
            inputs, targets, phase_index = self.task.generate_dataset(self.n_trials, **self.kwargs)
            
            # split
            train_inputs, test_inputs, train_targets, test_targets = train_test_split(
                inputs, targets, train_size=self.train_ratio, test_size=self.test_ratio
            )
            train_inputs, val_inputs, train_targets, val_targets = train_test_split(
                train_inputs, train_targets, train_size=self.train_ratio, test_size=self.val_ratio
            )
            
            # generate initial conditions
            if self.init_states is not None:
                assert(tinputs.shape[0] == self.init_states.shape[0])
                # split
                train_init_states, test_init_states = train_test_split(
                    self.init_states, train_size=self.train_ratio, test_size=self.test_ratio
                )

            else:
                train_init_states = torch.zeros_like(train_inputs)
            
            data = {
                'train_inputs': train_inputs,
                'train_targets': train_targets,
                'val_inputs': val_inputs,
                'val_targets': val_targets,
                'test_inputs': test_inputs,
                'test_targets': test_targets,
                'phase_index': phase_index
            }
            
            # save
            os.makedirs(self.data_dir, exist_ok=True)
            with h5py.File(self.dpath, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)
            logger.info(f"Dataset saved to {self.dpath}")
            
        

    def setup(self, stage=None):
        
        # load the saved h5py dataset
        with h5py.File(self.dpath, 'r') as f:
            self.train = TensorDataset(f['train_inputs'], f['train_targets'])
            self.val = TensorDataset(f['val_inputs'], f['val_targets'])
            self.test = TensorDataset(f['test_inputs'], f['test_targets'])
            self.phase_index = f['phase_index']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.kwargs['batch_size'])