from glob import glob
from config import Training_config, Database_config 
from monai.data import DataLoader, ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, RandSpatialCrop, RandRotate90
import os
from pathlib import Path
from monai.transforms import Transform
import numpy as np
from collections import defaultdict
import nibabel as nib


class RemoveChannels(Transform):
    def __init__(self, channels_to_remove=None):
        """
        Initialize the transformation
        """
        self.channels_to_remove = channels_to_remove

    def __call__(self, img):
        """
        Apply the transformation.

        Args:
            img (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Transformed image tensor with specified channels removed.
        """
        if self.channels_to_remove is None:
            return img
        channels_to_keep = [i for i in range(img.shape[0]) if i not in self.channels_to_remove]
        return img[channels_to_keep, ...]



def create_dataloader(
    val_size: int,
    images: list[Path],
    segs: list[Path],
    workers: int,
    train_batch_size: int,
    total_train_data_size: int,
    current_train_data_size: int,
    cropped_input_size: list,
    channels_to_remove: list,
    k_fold: dict,
    image_only: bool = False,
) -> None:
    """Create monai wrapped dataloaders for training and validation data"""

    if k_fold is None:
        div = total_train_data_size//current_train_data_size
        rem = total_train_data_size%current_train_data_size

    # training and validaiton split and index

    #k_fold
    if val_size == 0:
        raise ValueError(
            "Validation size must be greater than 0 for k-fold cross-validation."
        )

    if k_fold is not None:
        train_indices = k_fold["train"]
        val_indices = k_fold["val"]
        train_images = [images[i] for i in train_indices]
        train_segs = [segs[i] for i in train_indices]
        val_images = [images[i] for i in val_indices]
        val_segs = [segs[i] for i in val_indices]

    else:
        # standard set up from paper
        train_images = images[:-val_size]
        train_images = train_images * div + train_images[:rem]
        train_segs = segs[:-val_size]
        train_segs = train_segs * div + train_segs[:rem]
        val_images = images[-val_size:]
        val_segs = segs[-val_size:]
    # image augmentation through spatial cropping to size and by randomly rotating

    train_imtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
            RemoveChannels(channels_to_remove),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
            RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )

    seg_imtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
            RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
            RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )

    val_imtrans = Compose([EnsureChannelFirst(),RemoveChannels(channels_to_remove)])
    val_segtrans = Compose([EnsureChannelFirst()])
    # create a training data loader
    

    train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=seg_imtrans)
    ######################################################
    # Create a training data loader
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=0)

    # create a validation data loader
    val_ds = ImageDataset(
        val_images,
        val_segs,
        transform=val_imtrans,
        seg_transform=val_segtrans,
        image_only=image_only,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)

    # Create a custom dataset class to include the mean
    # class CustomImageDataset(ImageDataset):
    #     def __init__(self, *args, means, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         self.means = means

    #     def __getitem__(self, index):
    #         data = super().__getitem__(index)
    #         mean = self.means[index]
    #         return data[0], data[1], mean

    # Create a training data loader with original means


    # Attach the original means to the dataloaders
    
    return train_loader, val_loader


def get_dataloader(
    train_config: Training_config,
    database_config: Database_config,
    datasetlist: list[str],
    cropped_input_size: list[int],
    data_size: int,
    channels_copy: dict,
    k_fold: dict,
):
    """ Get the dataloader for the training and validation data for each dataset in the datasetlist"""

    # path initialization
    train_loaders = []
    # val_loaders = []
    val_loader = {}
    data_loader_map = {}
    img_path = database_config.img_path
    seg_path = database_config.seg_path

    # get dataloader
    for dataset in datasetlist:
        print("Training: ", dataset)
        val_size = database_config.total_size[dataset] - database_config.train_size[dataset]
        images = sorted(glob(os.path.join(img_path[dataset], "*.*")))
        segs = sorted(glob(os.path.join(seg_path[dataset], "*.*")))

        # select channels to remove from the dataset in question.
        channels_to_remove = get_modalities_drop(dataset, channels_copy[dataset], train_config.modality_remove)

        train_loader_one, val_loader[dataset] = create_dataloader(
            val_size=val_size,
            images=images,
            segs=segs,
            workers=train_config.workers,
            train_batch_size=train_config.train_batch_size,
            total_train_data_size=data_size,
            current_train_data_size=database_config.train_size[dataset],
            cropped_input_size=cropped_input_size,
            channels_to_remove=channels_to_remove,
            k_fold=k_fold,
        )
        data_loader_map[dataset] = len(train_loaders)
        train_loaders.append(train_loader_one)

        # TODO: can remove the val-loaders list i am pretty sure as does not get called later in script
        # val_loaders.append(val_loader[dataset])

    return train_loaders, val_loader, data_loader_map  


def get_modalities_drop(
    dataset: str, modalities_present: list[str], remove_modality: str
) -> list[int]:
    """Get the modalities to drop from the image tensor"""

    if remove_modality is None:
        return None

    if remove_modality in modalities_present and len(modalities_present) == 1:
        raise ValueError(
            f"Modality {remove_modality} is the only modality present in dataset {dataset}"
        )

    if remove_modality not in modalities_present:
        print(f"Modality {remove_modality} not present in dataset {dataset}")

    return [
        i
        for i, modality in enumerate(modalities_present)
        if modality == remove_modality
    ]


################# FOR INFERENCE SCRIPT ####################


def create_test_val_loader(
    val_size: int,
    images,
    segs,
    workers: int,
    dataset: str,
    modality_remove: str,
    channels: dict,
    image_only: bool = False,
):
    """Create monai wrapped dataloaders for validation data"""

    # image augmentation through spatial cropping to size and by randomly rotating

    channels_to_remove = get_modalities_drop(dataset, channels, modality_remove)

    val_imtrans = Compose([EnsureChannelFirst(), RemoveChannels(channels_to_remove)])
    val_segtrans = Compose([EnsureChannelFirst()])
    # create a training data loader

    # create a validation data loader
    val_ds = ImageDataset(
        images[-val_size:],
        segs[-val_size:],
        transform=val_imtrans,
        seg_transform=val_segtrans,
        image_only=image_only,
    )

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)

    return val_loader


########################################################################






if "__main__" == __name__:

    # TEST GET_MODALITIES DROP
    yo = get_modalities_drop("BRATS", ["T1", "T2", "T1ce", "FLAIR"], "FLAIR")
    print(yo)

    # TEST GET_DATALOADER
    train_config = Training_config()
    database_config = Database_config()
    datasetlist = ["BRATS", "ATLAS"]
    cropped_input_size = [128, 128, 128]
    data_size = 100
    channels_copy = {"BRATS": ["T1", "T2", "T1C", "FLAIR"], "ATLAS": ["T1"]}
    train_loaders, val_loader, data_loader_map = get_dataloader(
        train_config,
        database_config,
        datasetlist,
        cropped_input_size,
        data_size,
        channels_copy,
    )
