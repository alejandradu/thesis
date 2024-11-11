import torch
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from AbstractClass import SyntheticTask

# NOTE: generate docstrings with codeium, the period args are fractions of the trial

class CDM(SyntheticTask):
    """Represent a context-dependent decision making tasks
    for a fixed set of coherences"""
    
    hi = 1
    lo = -1

    def __init__(self,seed=0, coherences=None):
        super().__init__()
        self.seed = seed
        if coherences is None:
            self.coherences = [-4,-2,-1,1,2,4]
        else:
            self.coherences = coherences
        
    def generate_dataset(self, n_trials, bin_size=20, noise=0.1, n_timesteps=1370, fix=100, ctx=350, 
                       stim=800, mem=100, res=20, random_trials=False, ctx_choice=None,
                       coh_choice0=None, coh_choice1=None, coh_scale=1e-1, ctx_scale=1e-1):
        
        # NOTE: producess all trials of same total duration, the mask if for the loss
        # NOTE: not reducing bins yet
        
        if random_trials:
            ctx = np.random.randint(200 / bin_size, 600 / bin_size)
            stim = np.random.randint(200 / bin_size, 1600 / bin_size)
            mem = np.random.randint(200 / bin_size, 1600 / bin_size)
            res= np.random.randint(300 / bin_size, 700 / bin_size)
            
            # renormalize the indices
            total = ctx + stim + mem + res
            ctx = floor((ctx / total) * (n_timesteps - fix))
            stim = floor((stim / total) * (n_timesteps - fix))
            mem = floor((mem / total) * (n_timesteps - fix))
            res = floor((res / total) * (n_timesteps - fix))
            
        # get the timestamps for each period
        stim_begin = fix + ctx
        mem_begin = stim_begin + stim
        res_begin = mem_begin + mem
        
        # record timestamps when each phase begins
        if not random_trials:
            phase_index = {
                'fix': 0,
                'ctx': fix,
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
        inputs_sensory = noise * torch.randn((n_trials, n_timesteps, 2), dtype=torch.float32)
        inputs_context = torch.zeros((n_trials, n_timesteps, 2))
        inputs = torch.cat([inputs_sensory, inputs_context], dim=2)
        targets = torch.zeros((n_trials, n_timesteps, 1), dtype=torch.float32)
        mask = torch.zeros((n_trials, n_timesteps, 1), dtype=torch.float32)
            
        for n in range(n_trials):
 
            # recalculate time stamps
            if random_trials:
                ctx = np.random.randint(200, 600)
                stim = np.random.randint(200, 1600)
                mem = np.random.randint(200, 1600)
                res= np.random.randint(300, 700)

                # renormalize the fractions
                total = ctx + stim + mem + res
                ctx = floor((ctx / total) * (n_timesteps - fix))
                stim = floor((stim / total) * (n_timesteps - fix))
                mem = floor((mem / total) * (n_timesteps - fix))
                res = floor((res / total) * (n_timesteps - fix))
            
                stim_begin = fix + ctx
                mem_begin = stim_begin + stim
                res_begin = mem_begin + mem
                n_timesteps = res_begin + res
                
                phase_index['fix'][n] = 0
                phase_index['ctx'][n] = fix
                phase_index['stim'][n] = stim_begin
                phase_index['mem'][n] = mem_begin
                phase_index['res'][n] = res_begin
                
            if coh_choice0 is None:
                coh_choice0 = np.random.choice(self.coherences)
            if coh_choice1 is None: 
                coh_choice1 = np.random.choice(self.coherences)
                
            inputs[n, stim_begin:mem_begin, 0] += coh_choice0 * coh_scale
            inputs[n, stim_begin:mem_begin, 1] += coh_choice1 * coh_scale
            
            if ctx_choice is None:
                ctx_choice = np.random.choice([0,1])
                
            if ctx_choice == 0:
                inputs[n, fix:res_begin, 2] = 1 * ctx_scale
                targets[n, res_begin:, 0] = self.hi if coh_choice0 > 0 else self.lo
            elif ctx_choice == 1:
                inputs[n, fix:res_begin, 3] = 1 * ctx_scale
                targets[n, res_begin:, 0] = self.hi if coh_choice1 > 0 else self.lo

            mask[n, res_begin:, 0] = 1  # for the loss

        return inputs, targets, phase_index, mask
    
    
    def plot_trial(self, **kwargs):
        
        inputs, targets, phase_index, _= self.generate_dataset(1, **kwargs)
        inputs = inputs.squeeze().numpy() 
        targets = targets.squeeze().numpy()
        
        # plot the inputs and targets
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(inputs, label=['Channel 0', 'Channel 1', 'Context 0', 'Context 1'])
        ax[0].set_title("Inputs")
        ax[0].legend()
        ax[1].plot(targets)
        ax[1].set_title("Targets")
        
        # plot vertical dotted lines at x values of phase_index
        for key, val in phase_index.items():
            ax[0].axvline(x=val, color='k', linestyle='--')
            ax[1].axvline(x=val, color='k', linestyle='--')
        
        plt.show()
        
        
