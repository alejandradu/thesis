import torchmetrics
import torch

def loss_mse(output, target, mask):
    """
    Mean squared error loss 
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: idem
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    if mask is not None:
        loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    else:
        loss_tensor = (target - output).pow(2).mean(dim=-1)
    # Average over timesteps - account for batches of different size
    loss_by_batch = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    # Average over batches
    return loss_by_batch.mean()

def accuracy(output, target, mask, per_batch=False):
    """
    Return the accuracy of the model

    Args:
        output (tensor): predictions for outputs
        target (tensor): target for outputs
        mask (tensor): _description_
        per_batch: to return the accuracy per set of trials in a batch ("per trial")
                   returns (N, output_dimension). if output_dimension=1, done
                   might need more processing
    """
    if mask is not None:
        output = mask*output
    # TODO: generalize - no magic numbers
    global_func = torchmetrics.Accuracy(task='multiclass', num_classes=2)
    # apply this twice to get the per batch accuracy
    trial_func = torchmetrics.Accuracy(task='multiclass', num_classes=2, multidim_average='samplewise')
    
    if per_batch:
        return trial_func(output, target)
    else:
        return global_func(output, target)
    
    
def accuracy_raw(output, target, mask=None, per_batch=False):
    """
    Return the accuracy of the model

    Args:
        output (tensor): predictions for outputs
        target (tensor): target for outputs
        mask (tensor): mask to apply to the outputs
        per_batch (bool): to return the accuracy per set of trials in a batch ("per trial")
                          returns (N, output_dimension). if output_dimension=1, done
                          might need more processing
    """
    if mask is not None:
        output = mask * output

    # Get the predicted class by taking the argmax
    _, predicted = torch.max(output, dim=1)

    if per_batch:
        # Calculate accuracy per batch
        correct = (predicted == target).float()
        accuracy_per_batch = correct.mean(dim=1)
        return accuracy_per_batch
    else:
        # Calculate overall accuracy
        correct = (predicted == target).float().sum()
        total = target.numel()
        overall_accuracy = correct / total
        return overall_accuracy
