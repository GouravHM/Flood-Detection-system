import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import sys

# Import your U-Net
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.unet_model import UNET

# --- CONFIGURATION (Tuned for your Laptop) ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 2        # Small batch size
NUM_EPOCHS = 5        # Fast training
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_IMG_DIR = "data/train/images/"
TRAIN_MASK_DIR = "data/train/masks/"

class IndiaFloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        
        # Load Image (Assam Satellite)
        image = np.array(Image.open(img_path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
        
        # Load Mask (The red output, converted to Black/White)
        # We look for the Red channel to find the flood
        mask_rgba = np.array(Image.open(mask_path).convert("RGBA").resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
        
        # In your temp_mask.png, Flood is RED (255, 0, 0)
        # We create a binary mask: 1 where Red > 0, else 0
        mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.float32)
        mask[mask_rgba[:,:,0] > 0] = 1.0 

        # Convert to Tensor
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask

def train_model():
    print("⏳ Loading Assam Dataset...")
    images = os.listdir(TRAIN_IMG_DIR)
    
    if len(images) == 0:
        print("❌ Error: No images found. Did you do the Copy-Paste step?")
        return

    print(f"✅ Training on {len(images)} Assam images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Simple Training Loop
    model.train()
    dataset = IndiaFloodDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        loop = tqdm(loader, leave=True)
        for data, targets in loop:
            data = data.to(device)
            targets = targets.to(device)

            # Forward
            predictions = model(data)
            loss = loss_fn(predictions, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/trained_model.pth")
    print("\n🎉 Success! Trained exclusively on Assam Data.")

if __name__ == "__main__":
    train_model()