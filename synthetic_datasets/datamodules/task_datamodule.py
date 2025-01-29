import os
import h5py
import lightning as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# NOTE: you might want to add a seed, also retrieve
# phase_index_train and phase_index_val

class TaskDataModule(pl.LightningDataModule):
    """Organize data creation and saving/loading to train a 
    task-trained network for one tasks
    
    TODO: extend to multitask?
    TODO: add test split (for now take last val acc)
    
    Args:
        config: dict, all the hyperparameters for the task and data. Include:
            task: SyntheticTask task - PASS IT ALREADY CONFIGURED
            data_dir: str, directory for data saving, end WITH "/"
            init_states: if empty initializes to zero. Use for special inits of shape (hidden_size)
            # NOTE: need more complications to implement distribution initialization
            num_workers: int, match to number of CPUs per task
            train_ratio: float, ratio of training data
            val_ratio: float, ratio of validation data
            batch_size: int, size of the batches
    """
    
    # changing all hparams to a config dict 
    # config: task, data_dir, n_trials, batch_size, num_workers, train_ratio, val_ratio, init_states
    
    def __init__(self, config):
        
        super().__init__()
        self.task = config['task']
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.train_ratio = config['train_ratio']
        self.val_ratio = config['val_ratio']
        self.init_states = config['init_states']
        self.init_states_name = config['init_states_name']
        self.init_states_dimension = config['init_states_dimension']
        self.phase_index_train = None
        self.phase_index_val = None
        
        key = hash(frozenset(self.task.task_config.items()))
        self.dpath = os.path.join(self.data_dir, f"task{key}{self.init_states_name}.h5")
        

    # NOTE: should not assign states here
    def prepare_data(self):
        
        n_trials = self.task.n_trials
        idx = np.linspace(0, n_trials-1, n_trials-1).astype(int)
        
        # if data with same timing and splits already exists pass
        if os.path.exists(self.dpath):
            return
        else:
            inputs, targets, phase_index = self.task.generate_dataset()
            
            # expand the initial states to the number of n_trials
            if self.init_states is not None:
                init_states = np.tile(self.init_states, (n_trials, 1))
            else:
                init_states = np.zeros((n_trials, self.init_states_dimension))
            
            # split
            train_idx, val_idx = train_test_split(idx, train_size=self.train_ratio, test_size=self.val_ratio)
            
            # if the there is more than one phase info for trials
            if phase_index['fix'] > 1:
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
                'val_inputs': inputs[val_idx],
                'val_targets': targets[val_idx],
                'val_init_states': init_states[val_idx],
            }
            
            # save
            os.makedirs(self.data_dir, exist_ok=True)
            with h5py.File(self.dpath, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)
        

    # DO NOT REMOVE stage
    # BUG: should do splits here
    def setup(self, stage=None):
        
        # BUG: this is not finding the dataset (path is fine)
        # load the saved h5py dataset
        with h5py.File((self.dpath), 'r') as f:
            train_inputs = torch.tensor(f['train_inputs'][:])
            train_targets = torch.tensor(f['train_targets'][:])
            train_init_states = torch.tensor(f['train_init_states'][:])
            val_inputs = torch.tensor(f['val_inputs'][:])
            val_targets = torch.tensor(f['val_targets'][:])
            val_init_states = torch.tensor(f['val_init_states'][:])
            # phase_index_train = f['phase_index_train']
            # phase_index_val = f['phase_index_val']

        self.train_dataset = TensorDataset(train_inputs, train_targets, train_init_states)
        self.val_dataset = TensorDataset(val_inputs, val_targets, val_init_states)
        # self.phase_index_train = phase_index_train
        # self.phase_index_val = phase_index_val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def data_shape(self):
        """Return the length of the input and output vectors"""
        return self.train_dataset[0][0].shape[1], self.train_dataset[0][1].shape[1]
        