import lightning as pl
import torch 
import torch.nn as nn
import math
from models.metrics import loss_mse, accuracy

# NOTE: all models must have 
# forward(self, input, return_latents=, initial_states=):
# might want to take away the return_latents option

class GeneralModel(pl.LightningModule):
    """Wrap any neural net model for training and logging the same
    metrics with the same tuning config structure
    
    The params relevant for GeneralModel are lr and weight_decay, 
    but they must be passed before as part of the model_config"""
    
    def __init__(self, model_config: dict):
        super(GeneralModel, self).__init__()
        
        # create the model
        self.model_config = model_config
        self.model_class = model_config['model_class']
        self.lr = self.model_config['lr']
        self.weight_decay = self.model_config['weight_decay']
        
        # initialize the neural net (the model)
        self.model = self.model_class(model_config)
        
        # per epoch metrics
        self.eval_loss = []
        self.eval_accuracy = []
        
        
    def training_step(self, batch, batch_idx):
        # batch is the tensor dataset created in the datamodule
        inputs, targets, initial_states, mask = batch
        output, trajectories = self.model(inputs, return_latents=True, initial_states=initial_states)
        
        # adjust the mask dimensions - batch_size, n_timesteps, input_dim
        batch_size, n_timesteps, output_dim = output.shape

        loss = loss_mse(output, targets, mask)
        acc = accuracy(output, targets, mask)
        self.log('ptl/train_loss', loss, sync_dist=True)
        self.log('ptl/train_accuracy', acc, sync_dist=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        inputs, targets, initial_states, mask = batch
        output, trajectories = self.model(inputs, return_latents=True, initial_states=initial_states)
        
        # adjust the mask dimensions - batch_size, n_timesteps, input_dim
        batch_size, n_timesteps, output_dim = output.shape
        
        loss = loss_mse(output, targets, mask)
        acc = accuracy(output, targets, mask)
        self.eval_accuracy.append(acc)
        self.eval_loss.append(loss)
        return {"val_loss": loss, "val_accuracy": acc}
        
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.eval_loss).mean()
        avg_acc = torch.stack(self.eval_accuracy).mean()
        self.log("ptl/val_loss", avg_loss, sync_dist=True)
        self.log("ptl/val_accuracy", avg_acc, sync_dist=True)
        self.eval_loss.clear()
        self.eval_accuracy.clear()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    
class frRNN(nn.Module):
    """Implement Sompolinsky RNN, full rank"""
    
    def __init__(self, config):
        
        super(frRNN, self).__init__()
        self.config = config
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.noise_std = config['noise_std']  # note that this noise is for the evolution of the hidden states
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
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.wi_init = config['wi_init']
        self.si_init = config['si_init']
        self.wrec_init = config['wrec_init']
        self.b_init = config['b_init']
        self.wo_init = config['wo_init']
        self.so_init = config['so_init']
        
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
        # never optimize bias
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
        :param input: tensor of shape (batch_size, n_timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_latents: bool
        :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors for each trial
        :return: if return_latents=False, output tensor of shape (batch_size, #n_timesteps, output_dimension)
                 if return_latents=True, (output tensor, trajectories tensor of shape (batch_size, #n_timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        n_timesteps = input.shape[1]
    
        # initial state is a value for the hidden vector at time 0
        if initial_states is None:
            initial_states = self.h0
            #initial_states = initial_states.unsqueeze(0).expand(batch_size, self.hidden_size)

        # clone to keep reusability across trials and distributions
        h = initial_states.clone()
        r = self.non_linearity(initial_states).float()
        self._define_proxy_parameters()
        noise = torch.randn((batch_size, n_timesteps, self.hidden_size), device=self.wrec.device)
        output = torch.zeros((batch_size, n_timesteps, self.output_size), device=self.wrec.device)
        if return_latents:
            trajectories = torch.zeros((batch_size, n_timesteps+1, self.hidden_size), device=self.wrec.device)
            trajectories[:, 0, :] = h
            
        # TODO: set noise to zero, otherwise be careful if noise_std depends on alpha

        # simulation loop 
        for i in range(n_timesteps):
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.wrec.t()) + input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h + self.b).float()
            output[:, i, :] = self.output_non_linearity(h).float() @ self.wo_full
            if return_latents:
                trajectories[:, i + 1, :] = h           

        if return_latents:
            return output, trajectories
        else:
            return output

    
class lrRNN(nn.Module):
    """Implement Sompolinsky RNN, low rank"""
    
    def __init__(self, config):

        super(lrRNN, self).__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        self.noise_std = config['noise_std']
        self.alpha = config['alpha']
        self.rho = config['rho']
        self.train_wi = config['train_wi']
        self.train_wo = config['train_wo']
        self.train_h0 = config['train_h0']
        self.train_si = config['train_si']
        self.train_so = config['train_so']
        self.non_linearity = config['non_linearity']
        self.output_non_linearity = config['output_non_linearity']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.rank = config['rank']  
        self.wi_init = config['wi_init']
        self.si_init = config['si_init']
        self.b_init = config['b_init']
        self.wo_init = config['wo_init']
        self.so_init = config['so_init']
        self.m_init = config['m_init']
        self.n_init = config['n_init']

        # Define parameters
        self.wi = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.si = nn.Parameter(torch.Tensor(self.input_size))
        if self.train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False
        if not self.train_si:
            self.si.requires_grad = False
        # low rank approx matrix factors
        self.m = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.n = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))   
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))
        # never optimize bias
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
            if self.m_init is None:
                self.m.normal_()
            else:
                self.m.copy_(self.m_init)
            if self.n_init is None:
                self.n.normal_()
            else:
                self.n.copy_(self.n_init)
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
        :param input: tensor of shape (batch_size, #n_timesteps, input_dimension)
        Important: the 3 dimensions need to be present, even if they are of size 1.
        :param return_latents: bool
        :param initial_states: None or torch tensor of shape (batch_size, hidden_size) of initial state vectors for each trial if desired
        :return: if return_latents=False, output tensor of shape (batch_size, #n_timesteps, output_dimension)
                 if return_latents=True, (output tensor, trajectories tensor of shape (batch_size, #n_timesteps, #hidden_units))
        """
        batch_size = input.shape[0]
        n_timesteps = input.shape[1]
        if initial_states is None:
            initial_states = self.h0
        h = initial_states.clone()
        r = self.non_linearity(initial_states)
        self._define_proxy_parameters()
        noise = torch.randn(batch_size, n_timesteps, self.hidden_size, device=self.m.device)
        output = torch.zeros(batch_size, n_timesteps, self.output_size, device=self.m.device)
        if return_latents:
            # BUG: The expanded size of the tensor (10) must match the existing size (4) at non-singleton dimension 1.  Target sizes: [1, 10].  Tensor sizes: [335, 4]
            trajectories = torch.zeros(batch_size, n_timesteps + 1, self.hidden_size, device=self.m.device) 
            trajectories[:, 0, :] = h
            
        # TODO: set noise to zero, otherwise be careful if noise_std depends on alpha

        # simulation loop 
        for i in range(n_timesteps):
            # NOTE: why are we dividing by hidden_size?
            h = h + self.noise_std * noise[:, i, :] + self.alpha * (-h + r.matmul(self.n).matmul(self.m.t()) / self.hidden_size + input[:, i, :].matmul(self.wi_full))
            r = self.non_linearity(h + self.b)
            output[:, i, :] = self.output_non_linearity(h) @ self.wo_full
            if return_latents:
                trajectories[:, i + 1, :] = h           

        if return_latents:
            return output, trajectories
        
        return output
    

class nODE(nn.Module):
    """Imlement neural ordinary differential equation"""
    def __init__(self, config):

        super(nODE, self).__init__()
        self.num_layers = config['num_layers']
        self.hidden_size = config['hidden_size']  # this is the size that will plot
        self.latent_size = config['latent_size']  # this is the size of the function that parametrizes
        self.output_size = config['output_size']
        self.input_size = config['input_size']
        self.readout = config['output_mapping']
        self.generator = None
        # immediately initialize
        self.init_model(self.input_size, self.output_size)

    def init_model(self, input_size, output_size):
        self.generator = MLPCell(
            input_size, self.num_layers, self.hidden_size, self.latent_size
        )
        if self.readout is None:
            self.readout = nn.Linear(self.latent_size, output_size)
            # Initialize weights and biases for the readout layer
            nn.init.normal_(
                self.readout.weight, mean=0.0, std=0.01
            )  # Small standard deviation
            nn.init.constant_(self.readout.bias, 0.0)  # Zero bias initialization

    def forward(self, input, return_latents=False, initial_states=None):
        batch_size, n_timesteps, input_size = input.shape
        if initial_states is None:
            initial_states = torch.zeros((batch_size, self.latent_size), requires_grad=True, device=self.readout.device)
            
        output = torch.zeros(batch_size, n_timesteps, self.output_size, device=self.readout.device)
        hidden = initial_states.clone()
        
        if return_latents:
            trajectories = torch.zeros(batch_size, n_timesteps + 1, self.hidden_size, device=self.readout.device) 
            trajectories[:, 0, :] = hidden
            
        # simulation loop
        for i in range(n_timesteps):
            # input: batch_size, input_size ; hidden: batch_size, latent_size
            hidden = self.generator(input[:, i, :], hidden)
            output = self.readout(hidden)
            if return_latents:
                trajectories[:, i + 1, :] = hidden
        
        if return_latents:
            return output, trajectories
        
        return output

class MLPCell(nn.Module):
    """Parametrizes the NODE with an MLP"""
    def __init__(self, input_size, num_layers, layer_hidden_size, latent_size):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.layer_hidden_size = layer_hidden_size
        self.latent_size = latent_size
        layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size + latent_size, layer_hidden_size))
                layers.append(nn.ReLU())
            elif i == num_layers - 1:
                layers.append(nn.Linear(layer_hidden_size, latent_size))
            else:
                layers.append(nn.Linear(layer_hidden_size, layer_hidden_size))
                layers.append(nn.ReLU())
        self.vf_net = nn.Sequential(*layers)

    def forward(self, input, hidden):
        input_hidden = torch.cat([hidden, input], dim=1)
        return hidden + 0.1 * self.vf_net(input_hidden)