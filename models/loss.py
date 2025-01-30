
def loss_mse(output, target, mask):
    """
    Mean squared error loss 
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Average over timesteps - account for batches of different size
    loss_by_batch = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    # Average over batches
    return loss_by_batch.mean()