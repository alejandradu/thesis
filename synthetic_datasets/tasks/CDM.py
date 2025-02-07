import torch
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from synthetic_datasets.tasks.AbstractClass import SyntheticTask

# NOTE: generate docstrings with codeium, the period args are fractions of the trial

class CDM(SyntheticTask):
    """Represent a context-dependent decision making tasks
    for a fixed set of coherences
    
    Args:
        random_trials: bool, if True, the trial durations are randomly sampled"""
        
    # BUG: the tensors are of INCOMPLETE dimensions for the random_trials - also make the mask
    
    # NOTE: coherences here are possibly the same as Driscoll's targets
    
    hi = 1
    lo = -1

    def __init__(self, task_config):
        
        super().__init__()
        
        # save config dictionary to hash name of dataset
        self.task_config = task_config
        
        self.seed = task_config["seed"]
        self.coherences = task_config["coherences"]
        if self.coherences is None:
            self.coherences = [-4,-2,-1,1,2,4]

        self.input_size = 4   # 2 sensory inputs, 2 context inputs
        self.output_size = 1  # 1 decision output
        
        self.n_trials = task_config["n_trials"]
        self.bin_size = task_config["bin_size"]
        self.noise = task_config["noise"]
        self.n_timesteps = task_config["n_timesteps"]
        
        # these below lose importance if random_trials is True (most recent value)
        self.fix = task_config["fix"]
        self.ctx = task_config["ctx"]
        self.stim = task_config["stim"]
        self.mem = task_config["mem"]
        self.res = task_config["res"]
        self.random_trials = task_config["random_trials"]
        self.ctx_choice = task_config["ctx_choice"]
        self.coh_choice0 = task_config["coh_choice0"]
        self.coh_choice1 = task_config["coh_choice1"]
        self.coh_scale = task_config["coh_scale"]
        self.ctx_scale = task_config["ctx_scale"]
        
    def generate_dataset(self, to_plot=False):
        
        # NOTE: producess all trials of same total duration, the mask if for the loss
        # NOTE: not reducing bins yet
        # NOTE: time of initial fixation is always constant
        # NOTE: not returning mask, but mask later with phase_index
        # NOTE: not returning also de-noised "true" inputs yet
        
        # only generate one to plot
        if to_plot:
            n_trials = 1
        else:
            n_trials = self.n_trials
            
        # get the timestamps for each period
        if self.random_trials:
            
            fix = 100
            ctx = np.random.randint(200, 600)
            stim = np.random.randint(200, 1600)
            mem = np.random.randint(200, 1600)
            res= np.random.randint(300, 700)
            
            total = fix + ctx + stim + mem + res
    
            # rescale
            self.fix = floor(self.n_timesteps / self.bin_size * (fix / total))
            self.ctx = floor(self.n_timesteps / self.bin_size  * (ctx / total))
            self.stim = floor(self.n_timesteps / self.bin_size  * (stim / total))
            self.mem = floor(self.n_timesteps / self.bin_size  * (mem / total))
            self.res = floor(self.n_timesteps / self.bin_size  * (res / total))
                
        else:
            
            self.fix = floor(self.fix / self.bin_size)
            self.ctx = floor(self.ctx / self.bin_size)
            self.stim = floor(self.stim / self.bin_size)
            self.mem = floor(self.mem / self.bin_size)
            self.res = floor(self.res / self.bin_size)
            
        ctx_begin = self.fix
        stim_begin = ctx_begin + self.ctx
        mem_begin = stim_begin + self.stim
        res_begin = mem_begin + self.mem
        total_duration = res_begin + self.res
        
        # record timestamps when each phase begins
        if not self.random_trials:
            phase_index = {
                'fix': 0,
                'ctx': ctx_begin,
                'stim': stim_begin,
                'mem': mem_begin,
                'res': res_begin}
        else:
            phase_index = {
                'fix': np.zeros(n_trials, dtype=int),
                'ctx': np.zeros(n_trials, dtype=int),
                'stim': np.zeros(n_trials, dtype=int),
                'mem': np.zeros(n_trials, dtype=int),
                'res': np.zeros(n_trials, dtype=int)}
            
        # initialize the inputs: 2 noisy inputs drawn from the coherences
        # 2 contextual inputs as a one-hot encoding of the trial context
        inputs_sensory = self.noise * torch.randn((n_trials, total_duration, 2), dtype=torch.float32)
        inputs_context = torch.zeros((n_trials, total_duration, 2))
        inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
        targets = torch.zeros((n_trials, total_duration, 1), dtype=torch.float32)
        # mask for when all trials have the same total length and phase time stamps
        mask_not_random = torch.zeros((n_trials, total_duration, 1), dtype=torch.float32)
        # mark with ones only the response period
        mask_not_random[:, res_begin:, 0] = 1
        # for now return only the constant mask
        mask_random = torch.ones((n_trials, total_duration, 1), dtype=torch.float32)
            
        for n in range(n_trials):
 
            # recalculate time stamps
            if self.random_trials:
                fix = 100
                ctx = np.random.randint(200, 600)
                stim = np.random.randint(200, 1600)
                mem = np.random.randint(200, 1600)
                res= np.random.randint(300, 700)
                
                total = fix + ctx + stim + mem + res
        
                # rescale
                self.fix = floor(self.n_timesteps / self.bin_size * (fix / total))
                self.ctx = floor(self.n_timesteps / self.bin_size  * (ctx / total))
                self.stim = floor(self.n_timesteps / self.bin_size  * (stim / total))
                self.mem = floor(self.n_timesteps / self.bin_size  * (mem / total))
                self.res = floor(self.n_timesteps / self.bin_size  * (res / total))
                
                ctx_begin = self.fix
                stim_begin = ctx_begin + self.ctx
                mem_begin = stim_begin + self.stim
                res_begin = mem_begin + self.mem
                total_duration = res_begin + self.res
                
                phase_index['fix'][n] = 0
                phase_index['ctx'][n] = ctx_begin
                phase_index['stim'][n] = stim_begin
                phase_index['mem'][n] = mem_begin
                phase_index['res'][n] = res_begin
                
            if self.coh_choice0 is None:
                self.coh_choice0 = np.random.choice(self.coherences)
            if self.coh_choice1 is None: 
                self.coh_choice1 = np.random.choice(self.coherences)
                
            inputs[n, stim_begin:mem_begin, 0] += self.coh_choice0 * self.coh_scale
            inputs[n, stim_begin:mem_begin, 1] += self.coh_choice1 * self.coh_scale
            
            if self.ctx_choice is None:
                self.ctx_choice = np.random.choice([0,1])
                
            if self.ctx_choice == 0:
                inputs[n, ctx_begin:res_begin, 2] = 1 * self.ctx_scale
                targets[n, res_begin:, 0] = self.hi if self.coh_choice0 > 0 else self.lo
            elif self.ctx_choice == 1:
                inputs[n, ctx_begin:res_begin, 3] = 1 * self.ctx_scale
                targets[n, res_begin:, 0] = self.hi if self.coh_choice1 > 0 else self.lo
        
        if not self.random_trials:
            return inputs, targets, phase_index, mask_not_random
        else:
            return inputs, targets, phase_index, mask_random
    
    
    def plot_trial(self):
        """"Generate randomly one trial and plot it."""
        
        inputs, targets, phase_index = self.generate_dataset(to_plot=True)
        inputs = inputs.squeeze().numpy() 
        targets = targets.squeeze().numpy()
        
        # plot the inputs and targets
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(inputs, label=['Channel 0', 'Channel 1', 'Context 0', 'Context 1'])
        ax[0].set_title("Inputs")
        ax[0].set_xlim(0, len(inputs))
        ax[0].legend()
        ax[1].plot(targets)
        ax[1].set_title("Targets")
        
        # plot vertical dotted lines at x values of phase_index
        for key, val in phase_index.items():
            ax[0].axvline(x=val, color='k', linestyle='--')
            ax[1].axvline(x=val, color='k', linestyle='--')
        
        plt.show()
        
        
