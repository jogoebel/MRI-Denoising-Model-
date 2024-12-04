import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob

## Edited JQC 2024Nov27 - add ability to iterate through multiple modalities

# load all imgs given level, mode and modality
def load_images_from_directory(img_dirs, mode, modalities):
    images = []
    
    # iterate through all base paths for all noise levels
    for img_dir in img_dirs:
        for modality in modalities:
            directory = os.path.join(img_dir, modality, mode)
            print(f"Loading from directory: {directory}")
            
            #iterate through all .npy files
            for filename in sorted(os.listdir(directory)):
                if filename.endswith(".npy"):
                    file_path = os.path.join(directory, filename)
                    img = np.load(file_path)
                    images.append(img)

    # concatinate all imgs
    return np.concatenate(images, axis=0) if images else np.array([])

# edited to make it more generalizable: added mode, modality and increased level 
class MRIDataset(Dataset):
    def __init__(self, levels, modalities, mode="training",device=torch.device("cpu")):
        self.device = device  # Store device as an instance attribute

        base_paths = [f"./data/patchs32_32_{level}" for level in levels]

        # load free and noised MRI sets from multiple noise levels
        free_dirs = [os.path.join(base_path, "free") for base_path in base_paths]
        noised_dirs = [os.path.join(base_path, "noised") for base_path in base_paths]

        self.free_mri_set = load_images_from_directory(free_dirs, mode, modalities)
        self.noised_mri_set = load_images_from_directory(noised_dirs, mode, modalities)

        self.total = self.free_mri_set.shape[0]
        print(f"Loaded data shape: {self.free_mri_set.shape}")
        print(f"End reading {mode} dataset...")
           
    def __len__(self):
        return self.total

    def __getitem__(self, index):
        free_img = torch.from_numpy(self.free_mri_set[index]).float().to(self.device)
        noised_img = torch.from_numpy(self.noised_mri_set[index]).float().to(self.device)
        return {"free_img": free_img, "noised_img": noised_img}


# Example usage:
# dataset = MRIDataset(levels = [1,7,13], modality = ["IXI-PD","IXI-T1","IXI-T2"],mode="training", modality="IXI-PD")


def list_niigz_imgs(folder_path):
    """
    Loads all Nifti images from the specified folder.

    Parameters:
    - folder_path (str): Path to the folder containing the Nifti images.

    Returns:
    - file_list : List of images
    """
    
    # Search for all .nii.gz files in the specified folder
    file_pattern = os.path.join(folder_path, "*.nii.gz")
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"No .nii.gz files found in the folder: {folder_path}")
        return []

    return file_list

def list_npy_files(directory):
    """Returns a list of .npy file paths in the given directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
