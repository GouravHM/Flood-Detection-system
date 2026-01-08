import matplotlib.pyplot as plt
import torch

def save_results(pre, post, mask, save_path="outputs/flood_mask.png"):
    """
    Saves a side-by-side comparison for your report.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(pre)
    ax[0].set_title("Pre-Flood")
    ax[1].imshow(post)
    ax[1].set_title("Post-Flood")
    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title("Detected Flood Mask")
    
    for a in ax: a.axis('off')
    
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")