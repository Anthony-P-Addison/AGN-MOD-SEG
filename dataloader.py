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
from monai.transforms import LoadImaged
import torch
from monai.transforms import RandSpatialCropd, RandRotate90d


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
    
# def create_dataloader(
#     val_size: int,
#     images: list[Path],
#     segs: list[Path],
#     workers: int,
#     train_batch_size: int,
#     total_train_data_size: int,
#     current_train_data_size: int,
#     cropped_input_size: list,
#     channels_to_remove: list,
#     k_fold: dict,
#     image_only: bool = False,
# ) -> None:
#     """Create monai wrapped dataloaders for training and validation data"""

#     if k_fold is None:
#         div = total_train_data_size//current_train_data_size
#         rem = total_train_data_size%current_train_data_size

#     # training and validaiton split and index

#     #k_fold
#     if val_size == 0:
#         raise ValueError(
#             "Validation size must be greater than 0 for k-fold cross-validation."
#         )

#     if k_fold is not None:
#         train_indices = k_fold["train"]
#         val_indices = k_fold["val"]
#         train_images = [images[i] for i in train_indices]
#         train_segs = [segs[i] for i in train_indices]
#         val_images = [images[i] for i in val_indices]
#         val_segs = [segs[i] for i in val_indices]

#     elif k_fold is None:
#         train_images = images[:-val_size]
#         train_images = train_images * div + train_images[:rem]
#         train_segs = segs[:-val_size]
#         train_segs = train_segs * div + train_segs[:rem]
#         val_images = images[-val_size:]
#         val_segs = segs[-val_size:]


#     # image augmentation through spatial cropping to size and by randomly rotating

#     train_imtrans = Compose(
#         [
#             EnsureChannelFirst(strict_check=True),
#             RemoveChannels(channels_to_remove),
#             RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
#             # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
#             RandRotate90(prob=0.1, spatial_axes=(0, 2)),
#         ]
#     )

#     seg_imtrans = Compose(
#         [
#             EnsureChannelFirst(strict_check=True),
#             RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
#             # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
#             RandRotate90(prob=0.1, spatial_axes=(0, 2)),
#         ]
#     )

#     val_imtrans = Compose([EnsureChannelFirst(),RemoveChannels(channels_to_remove)])
#     val_segtrans = Compose([EnsureChannelFirst()])
#     # create a training data loader
    

#     train_ds = ImageDataset(train_images, train_segs, transform=train_imtrans, seg_transform=seg_imtrans)
#     ######################################################
#     # Create a training data loader
#     train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=0)

#     # create a validation data loader
#     val_ds = ImageDataset(
#         val_images,
#         val_segs,
#         transform=val_imtrans,
#         seg_transform=val_segtrans,
#         image_only=image_only,
#     )
#     val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)
    
#     return train_loader, val_loader
    





class TrainImageMaskDataset(ImageDataset):
    """
    Dataset that loads image, label, and mask files.
    Uses parent ImageDataset for image/label loading + initial transforms.
    Then applies a dictionary-based augmentation pipeline for synchronized random transforms.
    """
    def __init__(self, image_files, label_files, mask_files, 
                 transform=None, # Should be for initial img processing (EnsureChannelFirst, RemoveChannels)
                 seg_transform=None, # Should be for initial lbl processing (EnsureChannelFirst)
                 mask_transform=None, # For initial mask processing (EnsureChannelFirst)
                 # channels_to_remove is not directly used if RemoveChannels is in initial `transform`
                 init_crop_size=None, # Pass cropped_input_size here
                 dict_aug_prob=0.1): # Probability for RandRotate90d, example
        
        # Store file lists (mainly for mask loading)
        self.image_files = image_files
        self.label_files = label_files # Kept for consistency, though parent uses them
        self.mask_files = mask_files

        # Store the initial transform for the mask (e.g., EnsureChannelFirst)
        self.initial_mask_transform = mask_transform
        
        # --- Define the dictionary-based augmentation pipeline --- 
        dict_augmentations = []
        if init_crop_size:
            if not (isinstance(init_crop_size, (list, tuple)) and len(init_crop_size) == 3):
                raise ValueError("init_crop_size must be a list or tuple of 3 integers.")
            dict_augmentations.append(
                RandSpatialCropd(keys=["image", "label", "mask"], roi_size=tuple(init_crop_size), random_size=False)
            )
        
        dict_augmentations.append(
            RandRotate90d(keys=["image", "label", "mask"], prob=dict_aug_prob, spatial_axes=(0, 2))
        )
        # Add other dictionary-based augmentations here if needed
        self.dict_augment_pipeline = Compose(dict_augmentations)
        
        # Initialize parent class with image/label files and their *initial, non-random* transforms.
        # These `transform` and `seg_transform` should NOT contain RandSpatialCrop/RandRotate90.
        super().__init__(image_files, label_files, transform=transform, seg_transform=seg_transform)
        
        if len(mask_files) != len(image_files):
            raise ValueError(f"Got {len(mask_files)} masks but {len(image_files)} images")
        

    def __getitem__(self, index):
        # 1. Get image and label from parent class.
        # This applies the initial `transform` and `seg_transform` (e.g., EnsureChannelFirst, RemoveChannels).
        img, label = super().__getitem__(index)
        
        # 2. Load mask manually
        mask_path = self.mask_files[index]
        try:
            mask_nii = nib.load(str(mask_path))
            mask = torch.from_numpy(np.asarray(mask_nii.get_fdata(), dtype=np.float32))
        except Exception as e:
            print(f"Error loading mask file {mask_path} at index {index}: {e}")
            raise IOError(f"Error loading mask NIfTI file: {mask_path}") from e
        
        # 3. Apply initial (non-random) transform to mask
        # This should ensure the mask is a tensor and has a channel dimension (e.g., [C,H,W,D])
        if self.initial_mask_transform is not None:
            mask = self.initial_mask_transform(mask)
        else: # Fallback if no initial_mask_transform provided
            if mask.dim() == 3: mask = mask.unsqueeze(0) # Add channel dim if 3D
        
        # 4. Ensure all img, label, mask are 4D tensors [C,H,W,D] before dictionary transforms
        # The initial transforms passed to super() and initial_mask_transform should handle this.
        if not (isinstance(img, torch.Tensor) and img.ndim == 4):
            if isinstance(img, torch.Tensor) and img.ndim == 3: img = img.unsqueeze(0) 
            else: raise TypeError(f"Image at index {index} is not 3D or 4D tensor after initial_transform, shape: {img.shape if isinstance(img, torch.Tensor) else type(img)}")
        
        if not (isinstance(label, torch.Tensor) and label.ndim == 4):
            if isinstance(label, torch.Tensor) and label.ndim == 3: label = label.unsqueeze(0)
            else: raise TypeError(f"Label at index {index} is not 3D or 4D tensor after initial_transform, shape: {label.shape if isinstance(label, torch.Tensor) else type(label)}")

        if not (isinstance(mask, torch.Tensor) and mask.ndim == 4):
            if isinstance(mask, torch.Tensor) and mask.ndim == 3: mask = mask.unsqueeze(0)
            else: raise TypeError(f"Mask at index {index} is not 3D or 4D tensor after initial_mask_transform, shape: {mask.shape if isinstance(mask, torch.Tensor) else type(mask)}")

        # Ensure contiguity before putting into the dictionary for augmentation
        if isinstance(img, torch.Tensor): img = img.contiguous()
        if isinstance(label, torch.Tensor): label = label.contiguous()
        if isinstance(mask, torch.Tensor): mask = mask.contiguous()

        # 5. Create dictionary and apply dictionary-based augmentation pipeline
        data_dict = {"image": img, "label": label, "mask": mask}
        
        try:
            augmented_data_dict = self.dict_augment_pipeline(data_dict)
        except Exception as e:
            print(f"Error during dictionary augmentation for sample at index {index}: {e}")
            print(f"Data shapes before error: img: {img.shape}, label: {label.shape}, mask: {mask.shape}")
            # Include dtypes as well, as they can sometimes be relevant
            print(f"Data dtypes before error: img: {img.dtype}, label: {label.dtype}, mask: {mask.dtype}")
            raise RuntimeError(f"Dictionary augmentation pipeline failed at index {index}") from e

        # 6. Extract augmented tensors
        aug_img = augmented_data_dict["image"]
        aug_label = augmented_data_dict["label"]
        aug_mask = augmented_data_dict["mask"]
        
        return aug_img, aug_label, aug_mask


def create_dataloader(
    val_size: int,
    images: list[Path],
    segs: list[Path],
    masks: list[Path],
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

    # k_fold
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

    elif k_fold is None:
        #images
        train_images = images[:-val_size]
        train_images = train_images * div + train_images[:rem]
        train_segs = segs[:-val_size]
        train_segs = train_segs * div + train_segs[:rem]
        val_images = images[-val_size:]
        val_segs = segs[-val_size:]
        #masks
        val_masks = masks[-val_size:]
        train_masks = masks[:-val_size]
        train_masks = train_masks * div + train_masks[:rem]

    # image augmentation through spatial cropping to size and by randomly rotating

    train_imtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
            RemoveChannels(channels_to_remove),
            #RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
            #RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )

    seg_imtrans = Compose(
        [
            EnsureChannelFirst(strict_check=True),
            #RandSpatialCrop((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]), random_size=False),
            # RandCropByPosNegLabel((cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),label=train_segs),
            #RandRotate90(prob=0.1, spatial_axes=(0, 2)),
        ]
    )


    val_imtrans = Compose([EnsureChannelFirst(),RemoveChannels(channels_to_remove)])
    
    val_segtrans = Compose([EnsureChannelFirst()])

    # create a training data loader

    # train_ds = ImageDataset(
    #     train_images, train_segs, transform=train_imtrans, seg_transform=seg_imtrans
    # )

    train_ds = TrainImageMaskDataset(
    image_files=train_images,
    label_files=train_segs,
    mask_files=train_masks, 
    transform=train_imtrans,
    seg_transform=seg_imtrans,
    init_crop_size=cropped_input_size,
    )
    #mask_transform=mask_imtrans, 
    #channels_to_remove=channels_to_remove)


    ######################################################
    # Create a training data loader
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=0)

    # val_ds = ImageDataset(
    # image_files=val_images,
    # label_files=val_segs,
    # mask_files=val_masks, # Need to gather these mask file paths
    # transform=val_imtrans,
    # seg_transform=val_segtrans,
    # init_crop_size = cropped_input_size)
    # #mask_transform=mask_imtrans, # Define a transform for masks if needed
    # #channels_to_remove=channels_to_remove)


    # create a validation data loader
    val_ds = ImageDataset(
        val_images,
        val_segs,
        transform=val_imtrans,
        seg_transform=val_segtrans,
        image_only=image_only,
    )
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)

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
    mask_path = database_config.mask_path
    # get dataloader
    for dataset in datasetlist:
        print("Training: ", dataset)
        val_size = database_config.total_size[dataset] - database_config.train_size[dataset]
        images = sorted(glob(os.path.join(img_path[dataset], "*.*")))
        segs = sorted(glob(os.path.join(seg_path[dataset], "*.*")))
        masks = sorted(glob(os.path.join(mask_path[dataset], "*.*")))
        # select channels to remove from the dataset in question.
        channels_to_remove = get_modalities_drop(dataset, channels_copy[dataset], train_config.modality_remove)

        train_loader_one, val_loader[dataset] = create_dataloader(
            val_size=val_size,
            images=images,
            segs=segs,
            masks=masks,
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
    """Get the modalities to drop from the image tensor as list of integers"""

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
