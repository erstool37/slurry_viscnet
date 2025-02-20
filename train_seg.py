import torch
import wandb
import numpy as np
from statistics import mean
import glob
from preprocess.mobile_sam.utils import transforms
from torchvision import transforms as T
from torch.utils.data import Subset
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import namedtuple
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from torch.nn import functional as F

from preprocess.PreprocessorWebGL import PreprocessorWebGL
from preprocess.mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Test PyTorch and CUDA
print("PyTorch version:", torch.__version__)  # Should show CUDA version
print("CUDA version in PyTorch:", torch.version.cuda)  # Should match installed CUDA (e.g., 12.1)
print("CUDA available:", torch.cuda.is_available())  # Should return True
print("Number of GPUs:", torch.cuda.device_count())  # Should be > 0
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

# Model definition
BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_EPOCHS = 40
LR_RATE = 5e-4
CHECKPOINT = "preprocess/model_seg/Vortex0217_01.pth" # freshly trained model name
BASE_CHECKPOINT = "preprocess/model_seg/Vortex0217_01.pth" # pre-trained model name

CONFIG= {}

CONFIG["batch_size"] = BATCH_SIZE
CONFIG["learning_rate"] = LR_RATE
CONFIG["epochs"] = NUM_EPOCHS
CONFIG["architecture"] = "decoder finetuning"
CONFIG["dataset"] = "WebGL synthetic data"
CONFIG["scheduler"] = "CosineAnnealingLR"
CONFIG["loss"] = "DiceBCELoss"
CONFIG["checkpoint"] = CHECKPOINT

wandb.init(project="contour segmentation", reinit=True, resume="never", config= CONFIG)

# Preprocess images
# preprocessorWebGL = PreprocessorWebGL(radius_erosion=1, iter_erosion=1, radius_dilation=3, iter_dilation=2)
# preprocessorWebGL.process_images()

# Load images and pile in dataset
model_type = "vit_t"
sam_checkpoint = BASE_CHECKPOINT
device = "cpu"  # set device to cpu temporarily for dataset transforms

mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mobile_sam.to(device=device)

predictor = SamPredictor(mobile_sam)
mask_generator = SamAutomaticMaskGenerator(mobile_sam)

ImageMaskPathItem = namedtuple("ImageMaskPathItem", ["image_path", "mask_path"])
class ImageMaskDataset(Dataset):
    def __init__(self, root, image_subdir, mask_subdir, transform=None):
        super().__init__()
        self.root = root
        self.image_subdir = image_subdir
        self.mask_subdir = mask_subdir
        self.transform = transform

        images = glob.glob(osp.join(root, image_subdir, "*.png"))
        images = sorted(images)
        masks = [image.replace(image_subdir, mask_subdir) for image in images]

        self.path_items = [ImageMaskPathItem(image_path=image, mask_path=mask) for image, mask in zip(images, masks)]
        self._sanity_check()
        print(f"Found {len(self.path_items)} image-mask pairs")

    def _sanity_check(self):
        for item in self.path_items:
            assert osp.exists(item.image_path), f"Image path {item.image_path} does not exist"
            assert osp.exists(item.mask_path), f"Mask path {item.mask_path} does not exist"

    def __getitem__(self, index):
        item = self.path_items[index]
        image = np.array(Image.open(item.image_path).convert("RGB"))
        mask = np.array(Image.open(item.mask_path).convert("L"))

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask

    def __len__(self):
        return len(self.path_items)

transform = A.Compose(
    [A.Normalize(max_pixel_value=255.0), ToTensorV2()], is_check_shapes = False) #disabel size match verificaiton between img and mask

train_ds = ImageMaskDataset(root="dataset/WebGLfluid/processed_data", image_subdir="images", mask_subdir="masks", transform=transform)
val_ds = ImageMaskDataset(root="dataset/WebGLfluid/processed_data", image_subdir="images", mask_subdir="masks", transform=transform)

# Split dataset into train and validation
indices = np.arange(len(train_ds))
train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=True, random_state=42)

train_ds = Subset(train_ds, train_idx)
val_ds = Subset(val_ds, val_idx)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Training prerequisites
model_type = "vit_t"
sam_checkpoint = BASE_CHECKPOINT
device = "cuda" if torch.cuda.is_available() else "cpu"

mobile_sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint)
mobile_sam.to(device)
predictor = SamPredictor(mobile_sam)

# Freeze layers in MobileSAM
for name, param in mobile_sam.named_parameters():
    if name.startswith("image_encoder"): 
        param.requires_grad = False  # Freeze the vision encoder
    elif name.startswith("prompt_encoder"):
        param.requires_grad = False  # Freeze the prompt encoder

# Initialize the optimizer
optimizer = Adam(mobile_sam.mask_decoder.parameters(), lr=LR_RATE, weight_decay=0)
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
seg_loss = DiceBCELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Training loop
num_epochs = NUM_EPOCHS

mobile_sam.train()
for epoch in range(num_epochs):
    train_losses = []
    print(f"Epoch {epoch+1}/{num_epochs} - Training ")
    for batch in train_dl:
        # stacking mask/training model/mask retrieve
        images, masks = batch
        images = images.to(device)
        masks = (masks / 255).round().float().to(device) 
        batched_input = [{"image": img, "original_size": (256, 256)} for img in images]
        outputs = mobile_sam(batched_input=batched_input, multimask_output=False)
        predicted_masks = torch.stack([output["masks"].squeeze(1).float() for output in outputs])

        # loss calculation/optimization
        train_loss = seg_loss(predicted_masks, masks)
        train_losses.append(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # loss print
        if (len(train_losses)) % 50 == 0:
            mean_train_loss = mean(train_losses)
            wandb.log({"train_loss": mean_train_loss})
    train_losses.clear()

    # Validation loss calculation
    mobile_sam.eval()
    val_losses = []
    with torch.no_grad():
        print(f"Epoch {epoch+1}/{num_epochs} - Validation")
        for batch in val_dl:
            images, masks = batch
            images = images.to(device)
            masks = (masks / 255).round().float().to(device)
            batched_input = [{"image": img, "original_size": (256, 256)} for img in images]  
            outputs = mobile_sam(batched_input=batched_input, multimask_output=False)
            predicted_masks = torch.stack([output["masks"].squeeze(1).float() for output in outputs])
            val_loss = seg_loss(predicted_masks, masks)
            val_losses.append(val_loss.item())
    mean_val_loss = mean(val_losses)
    wandb.log({"val_loss": mean_val_loss})

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1}/{num_epochs} results - Train Loss: {mean_train_loss:.4f} Validation Loss: {mean_val_loss:.4f} - LR: {current_lr:.5f}")
wandb.finish()

# Save model
torch.save(mobile_sam.state_dict(), CHECKPOINT)