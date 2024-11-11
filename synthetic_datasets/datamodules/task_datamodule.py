import lightning as L
from torch.utils.data import random_split, DataLoader

class TaskDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = "./", **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.data_dir = data_dir

    # don't assign self. things here but in setup
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.kwargs['batch_size'])