import torch
import numpy as np





##### any defintions for augmentations go here ######

# TODO: move alpha (affects the data distribution to be selected from ) to the config ()
# TODO: look at cutting end of distribution s fof so never slecet image already available

def mixup_data(x:torch.tensor, alpha=2):
    """Returns mixed inputs, pairs of targets, and lambda
     alpha = 1 : uniformly sampled
     alpha>0 : normally distribution around poinst far from smaples"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        print("lam:", lam)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x[index[0]] + (1 - lam) * x[index[1]]
    #mixed_x = mixed_x.unsqueeze(0)
    return mixed_x, lam



