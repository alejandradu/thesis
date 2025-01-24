# NOTE: revise

def accuracy_general(output, targets, mask):
    good_trials = (targets != 0).any(dim=1).squeeze()
    target_decisions = torch.sign((targets[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze())
    decisions = torch.sign((output[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze())
    return (target_decisions == decisions).type(torch.float32).mean()