## saliency map for dealing with @torch.no_grad()

import torch







def calculate_contrib(net):

    mat = torch.zeros(1, 1, 36, 64)

    out = torch.zeros_like(mat)

    print(net(mat).shape)

    for i in range(36):

        for j in range(64):

            mat[0, 0, i, j] = 1

            out[0, 0, i, j] += torch.sum(net(mat))

            mat[0, 0, i, j] = 0

 

    out = out / out.max()

    return out

 

 

 

def conv(kernel, stride, padding, dilation):

    """

    Create a conv, with fixed weights and biases.

    The weights are all 1s, and the biases are all 0s.

    This means that we can use the conv to measure the effect of the input on each output pixel.

    """

    conv = torch.nn.Conv2d(1, 1, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation)

    conv.weight.data = torch.ones_like(conv.weight.data) / torch.numel(conv.weight.data)
    # sum of all weights is 1 rather than setting all weights to one

    conv.bias.data = torch.zeros_like(conv.bias.data)

    return conv
