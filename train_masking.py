import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp  # Importing the segmentation_models_pytorch library
import torch
from transformers import TrainingArguments, Trainer


# Custom Dataset Class for Segmentation
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Custom dataset for segmentation tasks.

        Args:
            image_dir (str): Directory containing the images.
            mask_dir (str): Directory containing the segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # Get all image filenames
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """Fetch an image and its corresponding mask."""
        # Get the image and mask file paths
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Open the image and mask using PIL
        image = Image.open(img_name).convert("RGB")  # Convert to RGB (3 channels)
        mask = Image.open(mask_name).convert("L")  # Convert to grayscale (1 channel)

        # Apply the transform (if any)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Parameters
IMAGE_SIZE = (640, 640)  # Resize images to a standard size
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
BATCH_SIZE = 8

# Directory paths where your images and masks are stored
image_dir = "/sise/home/oryanyeh/muze.ai/synthetic_dataset/images/train/"
mask_dir = "/sise/home/oryanyeh/muze.ai/synthetic_dataset/masks/train/"

# Define the transform (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Create dataset instance
dataset = SegmentationDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

# Create DataLoader instance
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load Pretrained U-Net Model from segmentation_models_pytorch
model = smp.Unet(
    encoder_name="resnet34",  # You can change the encoder (e.g., 'resnet50', 'efficientnet-b0', etc.)
    encoder_weights="imagenet",  # Use ImageNet weights for the encoder
    in_channels=3,  # 3 input channels (RGB)
    classes=6,  # Number of segmentation classes (you mentioned 6)
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Output directory
    evaluation_strategy="epoch",  # Evaluate after every epoch
    learning_rate=LEARNING_RATE,  # Learning rate
    per_device_train_batch_size=BATCH_SIZE,  # Batch size during training
    per_device_eval_batch_size=BATCH_SIZE,  # Batch size during evaluation
    num_train_epochs=NUM_EPOCHS,  # Number of training epochs
    weight_decay=0.01,  # Strength of weight decay
)

# Initialize Trainer with the model, arguments, and dataset
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # Training arguments
    train_dataset=dataset,  # Training dataset
    eval_dataset=dataset,  # Evaluation dataset
)

# Start the fine-tuning process
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Optionally, save the model after training
model.save_pretrained('./final_model')
