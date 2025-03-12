# 3D version of GIN in case you are using a 3D network. 3D ver. of IPA will be released soon
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace



##### 


class GradlessGCReplayNonlinBlock3D(nn.Module):
    def __init__(
        self,
        out_channel=32,
        in_channel=1,
        scale_pool=[1, 1],
        layer_id=0,
        use_act=True,
        requires_grad=False,
        init_scale="default",
        device_id: str = "1",
        **kwargs
    ):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock3D, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.layer_id = layer_id
        self.use_act = use_act
        self.requires_grad = requires_grad
        self.init_scale = init_scale
        self.cuda_id = "cuda:" + str(device_id)
        self.device = torch.device(self.cuda_id)
        
        assert requires_grad == False

    def forward(self, x_in, requires_grad=False):
        # random size of kernel
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,))
        k = self.scale_pool[idx_k[0]]

        nb, nc, nx, ny, nz = x_in.shape

        ker = torch.randn(
            [self.out_channel * nb, self.in_channel, k, k, k],
            requires_grad=self.requires_grad,
        ).to(self.device)
        shift = (
            torch.randn(
                [self.out_channel * nb, 1, 1, 1], requires_grad=self.requires_grad
            ).to(self.device)
            * 1.0
        )

        x_in = x_in.view(1, nb * nc, nx, ny, nz)
        x_conv = F.conv3d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=nb)
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny, nz)
        return x_conv


class GINGroupConv3D(nn.Module):
    def __init__(
        self,
        out_channel=3,
        in_channel=3,
        interm_channel=2,
        scale_pool=[1, 3],
        n_layer=4,
        out_norm="frob",
        init_scale="default",
        device_id: str = "1",
        **kwargs
    ):
        """
        GIN
        """

        self.cuda_id = "cuda:" + str(device_id)
        self.device = torch.device(self.cuda_id)
        
        super(GINGroupConv3D, self).__init__()
        self.scale_pool = (
            scale_pool  # don't make it tool large as we have multiple layers
        )
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel

        self.layers.append(
            GradlessGCReplayNonlinBlock3D(
                out_channel=interm_channel,
                in_channel=in_channel,
                scale_pool=scale_pool,
                init_scale=init_scale,
                layer_id=0,
            ).to(self.device)
        )
        for ii in range(n_layer - 2):
            self.layers.append(
                GradlessGCReplayNonlinBlock3D(
                    out_channel=interm_channel,
                    in_channel=interm_channel,
                    scale_pool=scale_pool,
                    init_scale=init_scale,
                    layer_id=ii + 1,
                ).to(self.device)
            )
        self.layers.append(
            GradlessGCReplayNonlinBlock3D(
                out_channel=out_channel,
                in_channel=interm_channel,
                scale_pool=scale_pool,
                init_scale=init_scale,
                layer_id=n_layer - 1,
                use_act=False,
            ).to(self.device)
        )

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        x_in = x_in.to(self.device)

        nb, nc, nx, ny, nz = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None, None]  # nb, 1, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1, 1).to(self.device)  # nb, nc, 1, 1

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == "frob":
            _in_frob = torch.norm(
                x_in.reshape(nb, nc, -1), dim=(-1, -2), p="fro", keepdim=False
            )
            _in_frob = _in_frob[:, None, None, None, None].repeat(1, nc, 1, 1, 1)
            _self_frob = torch.norm(
                mixed.view(nb, self.out_channel, -1),
                dim=(-1, -2),
                p="fro",
                keepdim=False,
            )
            _self_frob = _self_frob[:, None, None, None, None].repeat(
                1, self.out_channel, 1, 1, 1
            )
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed


########## unit test ########################################

if __name__ == "__main__":
    from pdb import set_trace
    import nibabel as nib

    # Load the NIfTI image
    

    nifti_file = '/home/magd6292/Documents/wentian_clone/MultiUnet/data/BRATS/Images/BRATS_001_normed_on_mask.nii.gz'
    nifti_img = nib.load(nifti_file)
    nifti_data = nifti_img.get_fdata()
    # Crop the nifti_data to a 128x64x64 crop
    nifti_data = np.transpose(nifti_data, (3, 0, 1, 2))
    nifti_data= nifti_data[0]
    crop_size = 128
    x, y, z = nifti_data.shape
    start_x = (x - crop_size) // 2
    start_y = (y - crop_size) // 2
    start_z = (z - crop_size) // 2
    nifti_data = nifti_data[
        start_x : start_x + crop_size,
        start_y : start_y + crop_size,
        start_z : start_z + crop_size,
    ]

    xin = torch.tensor(nifti_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # xin = torch.rand([2, 2, 64, 64, 32]).cuda()
    size = 1
    device_id = '1'
    augmenter = GINGroupConv3D(out_channel=size, in_channel=size,device_id = device_id)
    out = augmenter(xin)
    # set_trace()
    print(out.shape)

    ##########################################

    output_data = out.squeeze().cpu().numpy()
    import matplotlib.pyplot as plt

    # Plot the output data as an image
    plt.figure(figsize=(12, 6))

    # Plot the original data
    plt.subplot(1, 2, 1)
    plt.imshow(nifti_data[nifti_data.shape[0] // 2, :, :], cmap="gray")
    plt.imshow(output_data[output_data.shape[0] // 2, :, :], cmap="gray")
    plt.title("Output Image Slice")
    plt.axis("off")

    plt.savefig("comparison_image_slice.png")
    plt.imshow(output_data[output_data.shape[0] // 2, :, :], cmap="gray")
    plt.title("Output Image Slice")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.savefig("comparison_image_slice.png")
    plt.imshow(output_data[output_data.shape[0] // 2, :, :], cmap="gray")
    plt.title("Output Image Slice")
    plt.axis("off")
    plt.savefig("output_image_slice.png")

    output_img = nib.Nifti1Image(output_data, nifti_img.affine)

    # Save the output NIfTI image
    output_file = "save_segs/nifti"
    nib.save(output_img, output_file)
