from glob import glob
from config import Training_config, Database_config 
from monai.data import DataLoader, ImageDataset
from monai.transforms import Compose, EnsureChannelFirst, RandSpatialCrop, RandRotate90
import os
from pathlib import Path
from monai.transforms import Transform


class RemoveChannels(Transform):
    def __init__(self, channels_to_remove=None):
        """
        Initialize the transformation.

        Args:
            channels_to_remove (list[int]): List of channel indices to remove.
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



def create_dataloader(val_size:int, images:list[Path], segs: list[Path], workers:int, train_batch_size:int, total_train_data_size:int, current_train_data_size:int, cropped_input_size:list,channels_to_remove:list,image_only:bool = False) -> None:
    """Create monai wrapped dataloaders for training and validation data"""
    
    div = total_train_data_size//current_train_data_size
    rem = total_train_data_size%current_train_data_size

    # training and validaiton split and index

    train_images = images[:-val_size]
    train_images = train_images * div + train_images[:rem]
    train_segs = segs[:-val_size]
    train_segs = train_segs * div + train_segs[:rem]
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
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=0)
    # create a validation data loader
    val_ds = ImageDataset(images[-val_size:], segs[-val_size:], transform=val_imtrans, seg_transform=val_segtrans,image_only = image_only)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=workers, pin_memory=0)


    return train_loader, val_loader



def get_dataloader(train_config:Training_config, database_config:Database_config,datasetlist:list[str], cropped_input_size:list[int] , data_size:int,channels_copy:dict):
    """ Get the dataloader for the training and validation data for each dataset in the datasetlist"""
    
    # path initialization
    train_loaders = []
    #val_loaders = []
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
            channels_to_remove= channels_to_remove                                                                 
        )
        data_loader_map[dataset] = len(train_loaders)
        train_loaders.append(train_loader_one)

        #TODO: can remove the val-loaders list i am pretty sure as does not get called later in script
        #val_loaders.append(val_loader[dataset])

    return train_loaders, val_loader, data_loader_map  




def get_modalities_drop(dataset:str, modalities_present:list[str], drop_modality:str) -> list[int]:
    """Get the modalities to drop from the image tensor"""
    if drop_modality is None:
        return None

    if drop_modality in modalities_present and len(modalities_present) == 1:
        raise ValueError (f"Modality {drop_modality} is the only modality present in dataset {dataset}")

    if drop_modality not in modalities_present:
        print(f"Modality {drop_modality} not present in dataset {dataset}")
    
    return [i for i, modality in enumerate(modalities_present) if modality == drop_modality]
   