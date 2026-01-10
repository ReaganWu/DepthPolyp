import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import numpy as np
import glob
import random
import torch
from PIL import Image
import albumentations as Aug
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from setup.random_lightspot import AddLightSpots


# GLOBAL Config
SHUFFLE = True
NUM_WORKERS = 4
IMG_WIDTH = 224
IMG_HEIGHT = IMG_WIDTH
IMG_CHANNELS = 3
BATCH_SIZE = 16

def set_seed(seed=42):
    """fixed seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class KvasirDataset_v2(Dataset):
    def __init__(self, data, albumentations_transform = None, blurring_transform = None):
        self.data = data
        self.albumentations_transform = albumentations_transform
        self.blurring_albumentations = blurring_transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        # depth_mask = Image.open(depth_path).convert('RGB')
        if isinstance(mask, Image.Image):
            pass  
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask) 
        else:
            mask = Image.open(mask).convert('L')

        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']
        
        if self.blurring_albumentations:
            augmented_blur = self.blurring_albumentations(image=np.array(image))
            image_blur = augmented_blur['image']
            
        mask = torch.from_numpy(mask).unsqueeze(-1).float()

        return image_blur, image, mask

class PolypGenDataset(Dataset):
    def __init__(self, root_dir, transform=None, blurring_transform=None, selected_sequences=None):
        """
        Args:
            root_dir: PolypGen root
            transform: albumentations transform
            blurring_transform: blurring_transform, default is None. seq18~seq22 are the real-world blurry video.
            selected_sequences: the selected sequence for validating, e.g. ['seq18', 'seq20', 'seq21', 'seq22']
        """
        self.root_dir = root_dir
        self.transform = transform
        self.blurring_transform = blurring_transform
        self.data = []
        
        # 
        seq_folders = sorted([d for d in os.listdir(root_dir) if d.startswith('seq')])
        
        # ===== select the sequence =====
        if selected_sequences is not None:
            seq_folders = [s for s in seq_folders if s in selected_sequences]
            print(f"Selected sequences: {seq_folders}")
        
        for seq_folder in seq_folders:
            seq_path = os.path.join(root_dir, seq_folder)
            images_dir = os.path.join(seq_path, 'images')
            masks_dir = os.path.join(seq_path, 'masks')
            
            if os.path.exists(images_dir) and os.path.exists(masks_dir):
                image_files = sorted(os.listdir(images_dir), 
                                   key=lambda x: int(x.split('.')[0]))
                
                for img_file in image_files:
                    img_path = os.path.join(images_dir, img_file)
                    mask_path = os.path.join(masks_dir, img_file)
                    
                    if os.path.exists(mask_path):
                        self.data.append({
                            'image': img_path,
                            'mask': mask_path,
                            'sequence': seq_folder,
                            'frame': int(img_file.split('.')[0])
                        })
        
        print(f"PolypGen dataset loaded: {len(self.data)} frames from {len(seq_folders)} sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        mask = Image.open(item['mask']).convert('L')
        
        image = np.array(image)
        mask = np.array(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        if self.blurring_transform:
            augmented_blur = self.blurring_transform(image=image)
            image_blur = augmented_blur['image']
        else:
            image_blur = image
        
        # transfer and do the normalization
        image_blur = torch.from_numpy(image_blur).permute(2, 0, 1).float() / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        
        return image_blur, image, mask, item['sequence'], item['frame']

    
class BUSIDataset(Dataset):
    def __init__(self, data, albumentations_transform = None):
        self.data = data
        self.albumentations_transform = albumentations_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        # depth_mask = Image.open(depth_path).convert('RGB')
        if isinstance(mask, Image.Image):
            pass  
        elif isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
        else:
            mask = Image.open(mask).convert('L')  

        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=np.array(image), mask=np.array(mask))
            image = augmented['image']
            mask = augmented['mask']
        mask = torch.from_numpy(mask).unsqueeze(-1).float()
        return image, mask

def get_dataloader(dataset_name, batch_size=32, shuffle=True, num_workers=30):
    supported_datasets = {
        "KVASIR-v2": "vision@segmentation",
        "ColonDB-v2": "vision@segmentation",
        "ClinicDB-v2": "vision@segmentation",
        "PolypGen": "vision@segmentation",
    }
    # raise error if dataset not supported
    if dataset_name not in supported_datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets are {list(supported_datasets.keys())}")
    
    elif dataset_name == "KVASIR-v2":
        albumentations_transform = Aug.Compose([
            Aug.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),  # Add resize operation here
            Aug.Rotate(limit=15),  # rotation_range=0.1 (degree range)
            Aug.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),  # width_shift_range, height_shift_range, shear_range
            Aug.HorizontalFlip(p=0.3),  # horizontal_flip
            Aug.VerticalFlip(p=0.3),  # vertical_flip
            Aug.Affine(scale=(0.9, 1.1), p=1.0),  # fill_mode='constant' with scale range
        ], p=1.0)

        validation_transform = Aug.Compose([
            Aug.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),  # Add resize operation here
        ], p=1.0)

        blurring_transform = Aug.Compose([
            Aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2,0.2), p=1.0),  # contrast and brightness
            Aug.GaussianBlur(blur_limit=(3, 7), p=0.2),  # gaussian blurring
            Aug.MotionBlur(blur_limit=(3,29), p=1.0),         # motion blurring
            Aug.ImageCompression(quality_lower=30, quality_upper=70, p=0.5), # fake compression artifacts
            Aug.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.8, p=0.3),   # frog
            Aug.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3), # opticaldistortion
            AddLightSpots(radius_range=(5, 40), intensity=0.85, num_spots=1, always_apply=False, p=0.8),  # add light spots
        ], p=1.0)
        
        images_path = "/home/wangqj/workspace/endo/endoscope/dataset/Kvasir-SEG/images/*"
        masks_path = "/home/wangqj/workspace/endo/endoscope/dataset/Kvasir-SEG/masks/*"
        
        images = glob.glob(images_path)
        masks = glob.glob(masks_path)
        
        kvasir_data = [(img, mask) for img, mask in zip(images,masks)]
        train_data, val_data = train_test_split(kvasir_data, test_size=0.2, random_state=42)
        
        train_dataset = KvasirDataset_v2(data=train_data, albumentations_transform=albumentations_transform, blurring_transform=blurring_transform)
        val_dataset = KvasirDataset_v2(data=val_data, albumentations_transform=validation_transform, blurring_transform=blurring_transform)
        
        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=num_workers)
        val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=None, num_workers=num_workers)
        
        return train_data_loader, val_data_loader, dataset_name

    elif dataset_name == "ColonDB-v2":
        albumentations_transform = Aug.Compose([
            Aug.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),  # Add resize operation here
            Aug.Rotate(limit=15),  # rotation_range=0.1 (degree range)
            Aug.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),  # width_shift_range, height_shift_range, shear_range
            Aug.HorizontalFlip(p=0.3),  # horizontal_flip
            Aug.VerticalFlip(p=0.3),  # vertical_flip
            Aug.Affine(scale=(0.9, 1.1), p=1.0),  # fill_mode='constant' with scale range
        ], p=1.0)
        
        validation_transform = Aug.Compose([
            Aug.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),  # Add resize operation here
        ], p=1.0)

        blurring_transform = Aug.Compose([
            Aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2,0.2), p=1.0),  # contrast and brightness
            Aug.GaussianBlur(blur_limit=(3, 7), p=0.2),  # gaussian blurring
            Aug.MotionBlur(blur_limit=(15,29), p=1.0),         # motion blurring
            Aug.ImageCompression(quality_lower=30, quality_upper=70, p=0.5), # fake compression artifacts
            Aug.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.8, p=0.3),   # frog
            Aug.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3), # opticaldistortion
            AddLightSpots(radius_range=(5, 40), intensity=0.85, num_spots=1, always_apply=False, p=0.8),  # add light spots
        ], p=1.0)
        
        images_path = "/home/wuzy/DATASET/CVC-ColonDB/images/*"
        masks_path = "/home/wuzy/DATASET/CVC-ColonDB/masks/*"
        
        images = glob.glob(images_path)
        masks = glob.glob(masks_path)
        
        kvasir_data = [(img, mask) for img, mask in zip(images,masks)]
        train_data, val_data = train_test_split(kvasir_data, test_size=0.2, random_state=42)
        
        train_dataset = KvasirDataset_v2(data=train_data, albumentations_transform=albumentations_transform, blurring_transform=blurring_transform)
        val_dataset = KvasirDataset_v2(data=val_data, albumentations_transform=validation_transform, blurring_transform=blurring_transform)
        
        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=num_workers)
        val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=None, num_workers=num_workers)
        
        return train_data_loader, val_data_loader, dataset_name

    elif dataset_name == "ClinicDB-v2":
        albumentations_transform = Aug.Compose([
            Aug.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),  # Add resize operation here
            Aug.Rotate(limit=15),  # rotation_range=0.1 (degree range)
            Aug.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, p=0.5),  # width_shift_range, height_shift_range, shear_range
            Aug.HorizontalFlip(p=0.3),  # horizontal_flip
            Aug.VerticalFlip(p=0.3),  # vertical_flip
            Aug.Affine(scale=(0.9, 1.1), p=1.0),  # fill_mode='constant' with scale range
        ], p=1.0)
        
        validation_transform = Aug.Compose([
            Aug.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),  # Add resize operation here
        ], p=1.0)

        blurring_transform = Aug.Compose([
            Aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2,0.2), p=1.0),  # contrast and brightness
            Aug.GaussianBlur(blur_limit=(3, 7), p=0.2),  # gaussian blurring
            Aug.MotionBlur(blur_limit=(29,29), p=1.0),         # motion blurring
            Aug.ImageCompression(quality_lower=30, quality_upper=70, p=0.5), # fake compression artifacts
            Aug.RandomFog(fog_coef_lower=0.5, fog_coef_upper=0.8, p=0.3),   # frog
            Aug.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3), # opticaldistortion
            AddLightSpots(radius_range=(5, 40), intensity=0.85, num_spots=1, always_apply=False, p=0.8),  # add light spots
        ], p=1.0)
        
        images_path = "/home/wuzy/DATASET/ClinicDB/PNG/Original/*"
        masks_path = "/home/wuzy/DATASET/ClinicDB/PNG/GroundTruth/*"
        
        images = glob.glob(images_path)
        masks = glob.glob(masks_path)
        
        kvasir_data = [(img, mask) for img, mask in zip(images,masks)]
        train_data, val_data = train_test_split(kvasir_data, test_size=0.2, random_state=42)
        
        train_dataset = KvasirDataset_v2(data=train_data, albumentations_transform=albumentations_transform, blurring_transform=blurring_transform)
        val_dataset = KvasirDataset_v2(data=val_data, albumentations_transform=validation_transform, blurring_transform=blurring_transform)
        
        train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=num_workers)
        val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=None, num_workers=num_workers)
        
        return train_data_loader, val_data_loader, dataset_name

    elif dataset_name == "PolypGen":
        # 只做 resize，不做任何数据增强（因为是测试集）
        test_transform = Aug.Compose([
            Aug.Resize(height=IMG_HEIGHT, width=IMG_WIDTH, p=1.0),
        ], p=1.0)
        
        # 可选：测试模型对模糊的鲁棒性
        blurring_transform = Aug.Compose([
            Aug.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
            # Aug.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # Aug.MotionBlur(blur_limit=(3, 29), p=0.5),
            # Aug.ImageCompression(quality_lower=30, quality_upper=70, p=0.3),
        ], p=1.0)

        selected_seqs = ['seq18', 'seq19', 'seq20', 'seq21', 'seq22']

        polypgen_dataset = PolypGenDataset(
            root_dir="/home/wuzy/DATASET/polypgen",
            transform=test_transform,
            blurring_transform=None,  # 如果不需要模糊测试，设为 None
            selected_sequences=selected_seqs  # 只加载这些序列
        )
        
        # 推理数据集通常不需要 shuffle
        test_data_loader = DataLoader(
            polypgen_dataset, 
            batch_size=1, # it should be notice that the inference should be the video sequence (bs = 1)
            shuffle=False,  # keep same video streaming
            num_workers=num_workers
        )
        
        return test_data_loader, None, dataset_name  # Only Val


    print(f"Dataset {dataset_name} is loading")
    return train_data_loader, val_data_loader, dataset_name