import lightning as pl
import torch 
import torch.nn as nn
import math
from models.loss import loss_mse
from models.accuracy import accuracy_general

class tradRNN(pl.LightningModule):
    """Implement Sompolinsky RNN either full or low rank. Custom layer
    architecture and optimizer params"""
    
    # changing all the hparams for a config (dict?)
    # config: input_size, hidden_size, output_size, noise_std, alpha=0.2, rho=1,
    #         train_wi=False, train_wo=False, train_wrec=True, train_h0=False, train_si=True, train_so=True,
    #         wi_init=None, wo_init=None, wrec_init=None, si_init=None, so_init=None, b_init=None,
    #         add_biases=False, non_linearity=torch.tanh, output_non_linearity=torch.tanh, rank=128,
    #         lr=1e-3, weight_decay=0.0
    
    def __init__(self, config):

        super(tradRNN, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.noise_std = config['noise_std']
        self.alpha = config['alpha']
        self.rho = config['rho']
        self.train_wi = config['train_wi']
        self.train_wo = config['train_wo']
        self.train_wrec = config['train_wrec']
        self.train_h0 = config['train_h0']
        self.train_si = config['train_si']
        self.train_so = config['train_so']
        self.non_linearity = config['non_linearity']
        self.output_non_linearity = config['output_non_linearity']
        self.rank = config['rank']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.si = nn.Parameter(torch.Tensor(self.input_size))
        if self.train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not self.train_si:
            self.si.requires_grad = False
        self.wrec = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        if not self.train_wrec:
            self.wrec.requires_grad = False
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        if not self.add_biases:
            self.b.requires_grad = False
        self.wo = nn.Parameter(torch.Tensor(self.hidden_size, self.output_size))
        self.so = nn.Parameter(torch.Tensor(self.output_size))
        if self.train_wo:
            self.so.requires_grad = False
        if not self.train_wo:
            self.wo.requires_grad = False
        if not self.train_so:
            self.so.requires_grad = False
        self.h0 = nn.Parameter(torch.Tensor(self.hidden_size))
        if not self.train_h0:
            self.h0.requires_grad = False

        # Initialize parameters 
        # note that if you tried only the last layer, you would not use GD
        with torch.no_grad():
            if self.wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(self.wi_init)
            if self.si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(self.si_init)
            if self.wrec_init is None:
                self.wrec.normal_(std=self.rho / math.sqrt(self.hidden_size))
            else:
                self.wrec.copy_(self.wrec_init)
            if self.b_init is None:
                self.b.zero_()
            else:
                self.b.copy_(self.b_init)
            if self.wo_init is None:
                self.wo.normal_(std=1 / self.hidden_size)
            else:
                self.wo.copy_(self.wo_init)
            if self.so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(self.so_init)
            self.h0.zero_()
        self.wi_full, self.wo_full = [None] * 2
        self._define_proxy_parameters()

    def _define_proxy_parameters(self):
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so
        
    def forward(self, input, return_latents=False, initial_states=None):
        """
        :param input: tensor of shape (batch_size, #timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_latents: bool
        :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors for each trial if desired
        :return: if return_latents=False, output tensor of shape (batch_size, #timesteps, output_dimension)
                 if return_latents=True, (output tensor, trajectories tensor of shape (batch_size, #timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        seq_len = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(initial_states)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)
        if return_latents:
            trajectories = torch.zeros(batch_size, seq_len + 1, self.hidden_size, device=self.wrec.device)
            trajectories[:, 0, :] = h
            
        # TODO: set noise to zero, otherwise be careful if noise_std depends on alpha

        # simulation loop
        for i in range(seq_len):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.wrec.t()) + input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h + self.b)
            output[:, i, :] = self.output_non_linearity(h) @ self.wo_full

            if return_latents:
                trajectories[:, i + 1, :] = h

        if not return_latents:
            return output
        else:
            return output, trajectories
        
    def training_step(self, batch, batch_idx):
        inputs, targets, initial_states = batch
        output, trajectories = self(inputs, return_latents=True, initial_states=initial_states)
        # create mask to count only the response period
        mask = torch.ones_like(output) # TODO: fix mask
        loss = loss_mse(output, targets, mask)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        inputs, targets, initial_states = batch
        output, trajectories = self(inputs, return_latents=True, initial_states=initial_states)
        # create mask to count only the response period
        mask = torch.ones_like(output)
        loss = loss_mse(output, targets, mask)
        self.log('val_loss', loss)  
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)