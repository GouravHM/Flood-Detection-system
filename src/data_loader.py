import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FloodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pre_files = sorted(os.listdir(os.path.join(root_dir, "pre_flood")))
        self.post_files = sorted(os.listdir(os.path.join(root_dir, "post_flood")))

    def __len__(self):
        # We limit to the smallest number of files to avoid errors
        return min(len(self.pre_files), len(self.post_files))

    def __getitem__(self, idx):
        # Load Pre and Post images
        pre_path = os.path.join(self.root_dir, "pre_flood", self.pre_files[idx])
        post_path = os.path.join(self.root_dir, "post_flood", self.post_files[idx])
        
        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")

        if self.transform:
            pre_img = self.transform(pre_img)
            post_img = self.transform(post_img)

        # In a real scenario, we would also return a "Ground Truth" mask. 
        # For this urgent deadline, we return the images to train a reconstruction/difference model.
        return pre_img, post_img