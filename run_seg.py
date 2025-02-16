import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from create_data import preprocess_images  # Assuming this function is already available


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process reference and inspected images for defect detection.")

    # Add arguments for reference, inspected image, and the preprocessed input image
    parser.add_argument("--ref", help="Path to the reference image.", default=None)
    parser.add_argument("--insp", help="Path to the inspected image.", default=None)
    parser.add_argument("--input", help="Path to the image after preprocessing.", default=None)

    parser.add_argument("--save_dir", default="results/",
                        help="Directory to save the result masks (default: results/).")
    parser.add_argument("--weights", default="weights/yolo8s_seg.pt",
                        help="Path to the model weights (default: weights/yolo8s_seg.pt).")

    # Parse the arguments
    args = parser.parse_args()

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Check which inputs were provided
    if args.input:
        # If only preprocessed input image is provided
        input_image = cv2.imread(args.input)
        input_image = cv2.resize(input_image, (640, 640))
    elif args.ref and args.insp:
        # If both reference and inspected images are provided, preprocess them
        ref_image = cv2.imread(args.ref)
        ref_image = cv2.resize(ref_image, (640, 640))

        insp_image = cv2.imread(args.insp)
        insp_image = cv2.resize(insp_image, (640, 640))

        input_image = preprocess_images(ref_image, insp_image)
    else:
        print("Error: You must provide either '--input' or both '--ref' and '--insp'.")
        return


    # Load the model
    model_weights = args.weights
    model = YOLO(model_weights)  # Use your trained model's weight file

    # Run inference on the processed image
    results = model.predict(input_image)

    if results[0].masks:
        mask = torch.max(results[0].masks.data, dim=0)[0]
    else:  # Create a mask of size 640x640 if no mask is returned
        mask = torch.zeros(results[0].orig_img.shape[:2])

    # Convert the binary mask tensor to a NumPy array
    mask = mask.cpu().numpy()

    # Save the result as a PNG image
    mask_filename = "mask_" + os.path.basename(args.input if args.input else args.ref)
    mask_output_path = os.path.join(args.save_dir, mask_filename)
    image = Image.fromarray((mask * 255).astype(np.uint8))  # Convert 0/1 to 0/255
    image.save(mask_output_path)

    print(f"Mask saved to {mask_output_path}")


if __name__ == "__main__":
    main()
