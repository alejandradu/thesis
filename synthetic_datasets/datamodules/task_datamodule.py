import os
import h5py
import logging
import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from task_data.AbstractClass import SyntheticTask

# Configure logging
logger = logging.getLogger(__name__)

class TaskDataModule(pl.LightningDataModule):
    """Organize data creation and saving/loading to train a 
    task-trained network for one tasks
    
    TODO: extend to multitask?
    TODO: add test split (for now take last val acc)
    
    Args:
        task: SyntheticTask task to generate/handle the dataset
        data_dir: str, directory for data saving, end WITH "/"
        kwargs: timing parameters for the trials
        init_states: if empty initializes to zero. Use for special inits.
        num_workers: int, match to number of CPUs per task
    """
    def __init__(self, 
                 task: SyntheticTask,
                 data_dir: str = "./", 
                 n_trials: int = 1000,   # = n_samples for others
                 batch_size: int = 64, 
                 num_workers: int = 4, 
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.2,
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
        self.kwargs = kwargs
        self.init_states = init_states
        self.dpath = None
        self.phase_index_train = None
        self.phase_index_val = None

    # this will runk once: put complicated task generation here 
    def prepare_data(self):
        
        key = hash(frozenset(self.kwargs.items()))
        self.dpath = os.path.join(self.data_dir, f"task_{self.train_ratio}_{self.val_ration}_{key}.h5")
        idx = np.linspace(0, self.n_trials, self.n_trials).astype(int)
        
        # if data with same timing and splits already exists pass
        if os.path.exists(self.dpath):
            return
        else:
            logger.info(f"Creating dataset with: {self.kwargs}, \n{self.train_ratio}, {self.val_ratio}")
            inputs, targets, phase_index = self.task.generate_dataset(self.n_trials, **self.kwargs)
            # get initial conditions
            if self.init_states is not None:
                assert(inputs.shape[0] == self.init_states.shape[0])
                init_states = self.init_states
            else:
                init_states = torch.zeros_like(inputs)
            
            # split
            train_idx, val_idx = train_test_split(idx, train_size=self.train_ratio, test_size=self.val_ratio)
            
            # if the there is more than one phase info for trials
            if phase_index['fix'].shape[0] > 1:
                phase_index_train = {}
                phase_index_val = {}
                for key, value in phase_index.items():
                    phase_index_train[key] = value[train_idx]
                    phase_index_val[key] = value[val_idx]
            else:
                phase_index_train = phase_index
                phase_index_val = phase_index
   
            data = {
                'train_inputs': inputs[train_idx],
                'train_targets': targets[train_idx],
                'train_init_states': init_states[train_idx],
                'phase_index_train': phase_index_train,
                'val_inputs': inputs[val_idx],
                'val_targets': targets[val_idx],
                'val_init_states': init_states[val_idx],
                'phase_index_val': phase_index_val
            }
            
            # save
            os.makedirs(self.data_dir, exist_ok=True)
            with h5py.File(self.dpath, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)
            logger.info(f"Dataset saved to {self.dpath}")
        

    def setup(self, stage=None):
        
        # load the saved h5py dataset
        with h5py.File((self.dpath), 'r') as f:
            train_inputs = torch.tensor(f['train_inputs'][:])
            train_targets = torch.tensor(f['train_targets'][:])
            train_init_states = torch.tensor(f['train_init_states'][:])
            val_inputs = torch.tensor(f['val_inputs'][:])
            val_targets = torch.tensor(f['val_targets'][:])
            val_init_states = torch.tensor(f['val_init_states'][:])
            phase_index_train = f['phase_index_train']
            phase_index_val = f['phase_index_val']

        self.train_dataset = TensorDataset(train_inputs, train_targets, train_init_states)
        self.val_dataset = TensorDataset(val_inputs, val_targets, val_init_states)
        self.phase_index_train = phase_index_train
        self.phase_index_val = phase_index_val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)