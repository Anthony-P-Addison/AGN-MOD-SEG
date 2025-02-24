import torch
import numpy as np
import torchvision


##### any defintions for augmentations go here ######


def mixup_data(x: torch.tensor, all_mod_dropped: bool, one_mod_dropped:bool, two_not_dropped:bool, mod_3: bool):
    """Returns mixed inputs, pairs of targets, and lambda
    """

    if all_mod_dropped:
        # mixup data both dropped.
        #lam = np.random.uniform(0, 1)
        lam = np.random.beta(2,2)
        mixed_x = torch.mul(mix_operations(x[0]),lam) + torch.mul(mix_operations(x[1]),(1-lam))
    elif one_mod_dropped:
        # one mixup data dropped and other not dropped.
        #lam = np.random.uniform(0, 0.8)
        lam = np.random.beta(2,2)
        mixed_x = torch.mul(mix_operations(x[0]),lam) + torch.mul(mix_operations(x[1]),(1-lam))
    
    elif two_not_dropped:
        lam =np.random.beta(2,2)
        mixed_x = torch.mul(mix_operations(x[0]),lam) + torch.mul(mix_operations(x[1]),(1-lam))

    # mixing of three modalities only if they are all dropped. 
    elif mod_3:
        
        lam1, lam2, lam3 = np.random.dirichlet([2,2,2], size=1)[0]
        lam1, lam2, lam3 = float(lam1), float(lam2), float(lam3)
        mixed_x = torch.mul(mix_operations(x[0]),lam1) + torch.mul(mix_operations(x[1]),lam2) + torch.mul(mix_operations(x[2]),lam3)
        

    ###################################################################################
    # import matplotlib.pyplot as plt

    # # Assuming x is a batch of images with shape (batch_size, channels, height, width)
    # # Convert the tensor to a numpy array and transpose to (height, width, channels) for plotting
    # mixed_x_np = mixed_x.cpu().numpy()

    # mixed_x_slice = mixed_x_np[:, :, 64]

    # # Plot original images
    # x_np_0 = x[0].cpu().numpy()
    # x_np_1 = x[1].cpu().numpy()

    # if mod_3:
    #     x_np_2 = x[2].cpu().numpy()
    #     x_slice_2 = x_np_2[:, :, 64]

    # x_slice_0 = x_np_0[:, :, 64]
    # x_slice_1 = x_np_1[:, :, 64]

    # plt.figure(figsize=(12, 5))



    # plt.subplot(1, 4, 1)
    # plt.imshow(x_slice_0, cmap="gray")
    # plt.title("Original Image 1")
    # plt.axis("off")

    # plt.subplot(1, 4, 2)
    # plt.imshow(x_slice_1, cmap="gray")
    # plt.title("Original Image 2")
    # plt.axis("off")

    # plt.subplot(1, 4, 3)
    # if mod_3 == True:
    #     plt.imshow(x_slice_2, cmap="gray")
    # plt.title("Original Image 3")
    # plt.axis("off")


    # plt.subplot(1, 4, 4)
    # plt.imshow(mixed_x_slice, cmap="gray")
    # plt.title("Mixed Image")
    # plt.axis("off")
    # plt.show()
    # # plt.savefig('mixed_image.png')
    # # print("Image saved as 'mixed_image.png'")
    # plt.subplot(1, 4, 1)
    # plt.imshow(x_slice_0, cmap="gray")
    # plt.title("Original Image 1")
    # plt.axis("off")

    # plt.subplot(1, 4, 2)
    # plt.imshow(x_slice_1, cmap="gray")
    # plt.title("Original Image 2")
    # plt.axis("off")

    # plt.subplot(1, 4, 3)
    # if mod_3 == True:
    #     plt.imshow(x_slice_2, cmap="gray")
    # plt.title("Original Image 3")
    # plt.axis("off")


    # plt.subplot(1, 4, 4)
    # plt.imshow(mixed_x_slice, cmap="gray")
    # plt.title("Mixed Image")
    # plt.axis("off")
    # plt.show()
    # # plt.savefig('mixed_image.png')
    # print("Image saved as 'mixed_image.png'")

    return mixed_x





# Example usage in mix_operations
def mix_operations(
    x,
    translation=True,
    inversion=True,
    multiply = True,):
    """Apply mixup operations with optional translation, inversion, and contrast stretching.
    - Multiply
    - shift
    - invert"""

    if multiply:
        # Apply multiplication
        threshold = np.min(x)+0.01
        multiplicate = (np.random.uniform(0.9, 1.1))
        x = torch.where(x > threshold, x * multiplicate, x)
   
    if translation:
        translation = np.random.uniform(-0.3, 0.3)
        x = x + translation
    if inversion:
        # Apply inversion
        # invert_prob = torch.tensor(
        #     np.random.choice(
        #         [np.random.uniform(0.9, 1.1), np.random.uniform(-0.9, -1.1)]
        #     )
        # )
        threshold1 = translation + threshold
        
        invert_prob = torch.tensor(np.random.choice([1, -1]))
        
        x = torch.where(x > threshold1, x * invert_prob, x)
 
    return x




