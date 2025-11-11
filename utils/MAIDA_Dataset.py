import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Optional, Union, List, Callable

from utils.constants import ANNO_FILE_NAME_FIELD, ANNO_IMAGE_ID_FIELD
from utils.utils import get_image_file_path


class MAIDA_Dataset(Dataset):
    """Dataset class for handling MAIDA image data with metadata support."""
    
    def __init__(
        self, 
        data_path: Optional[str] = None, 
        data_source: Optional[str] = None, 
        image_meta: Optional[pd.DataFrame] = None, 
        dataset: Optional[Dataset] = None,
        transform: Optional[Callable] = None
    ):
        """Initialize the dataset from either metadata or another dataset.
        
        Args:
            data_path: Path to the data directory
            data_source: Source of the data (e.g., 'hospitals', 'train', 'test')
            image_meta: DataFrame containing image metadata. Note that even if the data_source has more 
                        images than the image_meta, only the images in the image_meta will be accessed.
            dataset: Another MAIDA_Dataset to copy from
            transform: Optional transform to apply to images
        """
        self.transform = transform or transforms.ToTensor()
        
        if dataset is not None:
            self._copy_from_dataset(dataset)
        elif image_meta is not None and data_path is not None and data_source is not None:
            self._init_from_metadata(data_path, data_source, image_meta)
        else:
            raise ValueError("Either provide dataset or (data_path, data_source, image_meta)")
  
    def _copy_from_dataset(self, dataset: Dataset) -> None:
        """Copy data from another dataset instance."""
        self.image_ids = dataset.image_ids
        self.id_to_path = dataset.id_to_path
        self.image_meta = dataset.image_meta
        
    def _init_from_metadata(self, data_path: str, data_source: str, image_meta: pd.DataFrame) -> None:
        """Initialize dataset from metadata."""
        self.image_ids = image_meta[ANNO_IMAGE_ID_FIELD].tolist()
        self.id_to_path = (
            image_meta.set_index(ANNO_IMAGE_ID_FIELD)[ANNO_FILE_NAME_FIELD]
            .apply(lambda f: get_image_file_path(data_path, data_source, f))
            .to_dict()
        )
        self.image_meta = image_meta

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int]]:
        """Get an image and its ID by index.
        
        Args:
            index: Index of the image
            
        Returns:
            Dict containing the image tensor and image ID
        """
        image_id = self.image_ids[index]
        image = self._load_image(image_id)
        
        return {
            "image": image,
            "image_id": image_id,
        }
    
    def _load_image(self, image_id: int) -> torch.Tensor:
        """Load and transform an image by its ID.
        
        Args:
            image_id: ID of the image to load
            
        Returns:
            Tensor representation of the image
        """
        image_path = self.id_to_path[image_id]
        image = Image.open(image_path)
        return self.transform(image)

    def reset_image_meta(self, image_meta: pd.DataFrame) -> None:
        """Update the dataset with new image metadata while preserving paths.
        
        Args:
            image_meta: New image metadata DataFrame
        """
        self.image_meta = image_meta
        self.image_ids = image_meta[ANNO_IMAGE_ID_FIELD].tolist()
        # Only keep paths for images that are in the new metadata
        self.id_to_path = {id: self.id_to_path[id] for id in self.image_ids if id in self.id_to_path}
    
    def get_image_meta(self) -> pd.DataFrame:
        """Get the image metadata.
        
        Returns:
            DataFrame containing image metadata
        """
        return self.image_meta
    
    def get_image_by_image_id(self, image_id: int) -> torch.Tensor:
        """Get an image by its ID.
        
        Args:
            image_id: ID of the image to load
            
        Returns:
            Tensor representation of the image
        """
        return self._load_image(image_id)