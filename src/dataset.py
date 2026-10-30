import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_test_transform

class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir)) if os.path.exists(root_dir) else []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []
        
        if os.path.exists(root_dir):
            for cls_name in self.classes:
                cls_dir = os.path.join(root_dir, cls_name)
                if os.path.isdir(cls_dir):
                    for img_name in os.listdir(cls_dir):
                        if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                            self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def prepare_data_loaders(data_dir, batch_size=32):
    train_trans, val_test_trans = get_transforms()
    full_dataset = PlantVillageDataset(data_dir)
    total_size = len(full_dataset)
    
    if total_size == 0:
        return None, None, None, []
        
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size], generator=generator)
    
    train_set.dataset.transform = train_trans
    val_set.dataset.transform = val_test_trans
    test_set.dataset.transform = val_test_trans
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, full_dataset.classes
