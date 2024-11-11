from abc import ABC, abstractmethod

    # below define the minimum attributes, methods, for all tasks
    
    # VALENTE TASK GENERATION METHODS
    # setup for globals
    # generate dataset, returns tensors
    # generate ordered inputs
    # generate dataset from conditions
    # accuracy
    # test (for loss and acc)
    # psychometric matrix
    
    # CTD TASK GENERATION METHODS
    # set seed
    # generate dataset, returns dict (for multitask, many tasks)
    # generate_trial
    # plot_tral
    # reset (not implemented)
    # step (not implemented)

class SyntheticTask(ABC):
    
    @abstractmethod
    def __init__(self):
        self.seed = None
    
    @abstractmethod
    def generate_dataset(self, n_trials, bin_size, noise):
        pass
    
    @abstractmethod
    def plot_trial(self):
        pass