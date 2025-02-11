from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
from create_data import *
from tqdm import tqdm
from PIL import Image
import torch

IMAGE_SIZE = (640, 640)
def evaluate_yolov5(model, ref_image_path, ins_image_path=None, save_dir="results/"):

    # load the images
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.resize(ref_image, IMAGE_SIZE)


    if ins_image_path is not None:
        ins_image = cv2.imread(ins_image_path)
        ins_image = cv2.resize(ins_image, IMAGE_SIZE)
        # preprocess the pair of images
        combined_image = preprocess_images(ref_image, ins_image)
    else:
        combined_image = ref_image

    # Run inference on the single image
    results = model.predict(combined_image)

    # Save the result image with bounding boxes drawn on it
    if ins_image_path is None:
        output_path = os.path.join(save_dir, os.path.basename(ref_image_path))
        output_path_mask = os.path.join(save_dir, "binary_mask_" + os.path.basename(ref_image_path))

    else:
        output_path = os.path.join(save_dir, os.path.basename(ins_image_path))
        output_path_mask = os.path.join(save_dir, "binary_mask_" + os.path.basename(ins_image_path))

    results[0].save(output_path)  # Save the result to the specified path

    if results[0].masks:
        mask = torch.max(results[0].masks.data, dim=0)[0]
    else: # crate mask wof 640x640
        mask = torch.zeros(results[0].orig_img.shape[:2])
    # Convert the binary mask tensor to a NumPy array
    mask = mask.cpu().numpy()

    # Save the result as a PNG image
    image = Image.fromarray((mask * 255).astype(np.uint8))  # Convert 0/1 to 0/255
    image.save(output_path_mask)


if __name__ == "__main__":
    pairs_image_paths = [
        ("data/defective_examples/case1_reference_image.tif", "data/defective_examples/case1_inspected_image.tif"),
        ("data/defective_examples/case2_reference_image.tif", "data/defective_examples/case2_inspected_image.tif"),
        ("data/non_defective_examples/case3_reference_image.tif", "data/non_defective_examples/case3_inspected_image.tif"),
        # ("data/other_dataset_exmple_reference.png", "data/other_dataset_exmple_inspected_1.png"),
        # ("data/other_dataset_exmple_reference.png", "data/other_dataset_exmple_inspected_2.png")
        ]

    pairs_image_out_paths = [("data/out_of_distrbution_data/1_ref.jpg", "data/out_of_distrbution_data/1_ins.jpg"),
                             ("data/out_of_distrbution_data/2_ins.jpg", "data/out_of_distrbution_data/2_ref.jpg"),
                             ("data/out_of_distrbution_data/3_1_ins.jpg", "data/out_of_distrbution_data/3_ref.jpg"),
                             ("data/out_of_distrbution_data/3_2_ins.jpg", "data/out_of_distrbution_data/3_ref.jpg"),
                             ]

    # samples from training data
    images_paths = [
        ("/sise/home/oryanyeh/muze.ai/synthetic_dataset/images/test/image_0.png", None), # 4
        ("/sise/home/oryanyeh/muze.ai/synthetic_dataset/images/test/image_100.png", None), # 1
        ("/sise/home/oryanyeh/muze.ai/synthetic_dataset/images/test/image_1005.png", None),  # 0
        ("/sise/home/oryanyeh/muze.ai/synthetic_dataset/images/test/image_1046.png", None),  # 3
        ("/sise/home/oryanyeh/muze.ai/synthetic_dataset/images/test/image_1047.png", None) # 2
    ]
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Load the trained model
    # model_weights = "weights/yolo5s_detect.pt"
    model_weights = "weights/yolo8s_seg.pt"
    model = YOLO(model_weights)  # Use your trained model's weight file (yolo5s_detect.pt)

    for reference_image_path, inspected_image_path in tqdm(pairs_image_paths):

        evaluate_yolov5(model, reference_image_path, inspected_image_path, save_dir=save_dir)
