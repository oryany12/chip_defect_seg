import os
import random

import cv2
import numpy as np
from PIL import Image
from skimage.draw import line
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
from tqdm import tqdm

# Parameters
NUM_IMAGES = 5000  # Number of images to generate per clean chip
IMAGE_SIZE = (640, 640)  # Resize images to a standard size

# Defect probabilities
defect_probabilities = {
    0: 0.20,
    1: 0.40,
    2: 0.25,
    3: 0.15
}

# defect_probabilities = {
#     0: 0.0,
#     1: 0.0,
#     2: 0.0,
#     3: 1.0
# }


def preprocess_images(ref_image, insp_image):
    ref_gray = np.mean(np.array(ref_image), axis=-1, keepdims=True).astype(np.uint8)  # Convert to grayscale
    insp_gray = np.mean(np.array(insp_image), axis=-1, keepdims=True).astype(np.uint8)
    diff = np.abs(ref_gray - insp_gray)  # Absolute difference

    # Stack grayscale and difference images into 3-channel RGB-like image
    combined_image = np.concatenate([ref_gray, insp_gray, diff], axis=-1)

    return combined_image


def augment_image(image, image_type='reference'):
    """Apply random augmentations to the image."""
    h, w = image.shape[:2]

    if image_type == 'reference':
        flip_prob = 0.5
        angle = random.randint(-180, 180)
        shift_x = random.randint(-int(0.2 * w), int(0.2 * w))
        shift_y = random.randint(-int(0.2 * h), int(0.2 * h))
        center_x, center_y = w // 2 + shift_x, h // 2 + shift_y
        zoom_factor = random.uniform(1.0, 1.3)

    elif image_type == 'inspected':
        flip_prob = 0
        angle = random.randint(-3, 3)
        shift_x = random.randint(-int(0.10 * w), int(0.10 * w))
        shift_y = random.randint(-int(0.10 * h), int(0.10 * h))
        center_x, center_y = w // 2 + shift_x, h // 2 + shift_y
        zoom_factor = random.uniform(1.0, 1.10)

    else:
        raise ValueError("Invalid image type! Choose 'reference' or 'inspected'.")

    # Flip
    if random.random() > 1 - flip_prob:
        image = cv2.flip(image, 1)

    # Apply translation (shifting the center of the image)
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    image = cv2.warpAffine(image, translation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Calculate the valid region to crop after the translation
    valid_x_min = max(0, shift_x)
    valid_y_min = max(0, shift_y)
    valid_x_max = min(w, w + shift_x)
    valid_y_max = min(h, h + shift_y)

    # Crop the image to remove empty spaces created by the shift
    image = image[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

    # Rotate the image around the new center
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Resize back to IMAGE_SIZE if needed
    image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))

    # zoom
    new_h, new_w = int(IMAGE_SIZE[0] * zoom_factor), int(IMAGE_SIZE[1] * zoom_factor)
    image = cv2.resize(image, (new_w, new_h))
    crop_h, crop_w = (new_h - IMAGE_SIZE[0]) // 2, (new_w - IMAGE_SIZE[1]) // 2
    image = image[crop_h:crop_h + IMAGE_SIZE[0], crop_w:crop_w + IMAGE_SIZE[1]]

    # Histogram adjustment
    gamma = random.uniform(0.8, 1.2)
    image = adjust_gamma(image, gamma)

    # Add grayscale noise
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create noise for dark and bright regions
    dark_noise = np.random.normal(0, random.randint(5, 15), image.shape[:2]).astype(np.int16)
    bright_noise = np.random.normal(0, random.randint(5, 15), image.shape[:2]).astype(np.int16)

    # Create a mask for dark and bright areas
    brightness_threshold = random.randint(75, 175)
    dark_mask = gray_image < brightness_threshold  # Dark areas (lower intensity)
    bright_mask = gray_image >= brightness_threshold  # Bright areas (higher intensity)

    # Apply dark noise & bright noise to dark regions
    noisy_image = gray_image.astype(np.int16) + np.where(dark_mask, dark_noise, 0)
    noisy_image = np.where(bright_mask, noisy_image + bright_noise, noisy_image)

    # Clip the pixel values to valid range (0-255)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Merge the noisy grayscale image back into a 3-channel image
    image = cv2.merge([noisy_image, noisy_image, noisy_image])

    return image


def random_defect_type():
    """Randomly select a defect type."""
    # defects_list = ["holy_white"]
    defects_list = ["scratch", "golden_bumps", "ring", "gate_pollution", "contamination", "holy_white"]
    index = random.choices(range(len(defects_list)))[0]
    return index, defects_list[index]


def inject_defect(image, defect_type):
    """Inject a defect into the image and return the bounding box."""
    img = image.copy()
    mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)  # Initialize the mask (black background)
    h, w = img.shape[:2]
    bounding_box = None

    if defect_type == "scratch":
        # Generate a random scratch line with a slight curve
        length = random.randint(40, 50)  # Length of the scratch
        start_x, start_y = random.randint(0, w - 1), random.randint(0, h - 1)
        curve_probability = 0.7  # Probability of having a curve
        if random.random() < curve_probability:
            # Add a very small curve
            control_x = (start_x + random.randint(-length * 2, length * 2))  # Slight horizontal offset
            control_y = (start_y + random.randint(-length * 2, length * 2))  # Slight vertical offset
            end_x = start_x + random.randint(-length, length)  # Randomize the x-coordinate of the end point
            end_y = start_y + random.randint(-length, length)  # Randomize the y-coordinate of the end point

            # Generate Bézier curve points
            # num_points = max(length * 5, 20)  # Increase points for small lengths
            t = np.linspace(0, 1, length)
            rr = ((1 - t) ** 2 * start_y + 2 * (1 - t) * t * control_y + t ** 2 * end_y).astype(np.int32)
            cc = ((1 - t) ** 2 * start_x + 2 * (1 - t) * t * control_x + t ** 2 * end_x).astype(np.int32)
        else:
            # Straight scratch
            end_x, end_y = random.randint(0, w - 1), random.randint(0, h - 1)
            rr, cc = line(start_y, start_x, end_y, end_x)
            rr, cc = rr[:length], cc[:length]  # Limit scratch length

        # Ensure the line is continuous by connecting adjacent points
        continuous_rr, continuous_cc = [], []
        for i in range(len(rr) - 1):
            segment_rr, segment_cc = line(rr[i], cc[i], rr[i + 1], cc[i + 1])
            continuous_rr.extend(segment_rr)
            continuous_cc.extend(segment_cc)

        # Convert to arrays
        continuous_rr = np.array(continuous_rr)
        continuous_cc = np.array(continuous_cc)

        # Uniform width for the scratch with reduced thickness
        max_thickness = 1  # Maximum thickness reduced to 1 pixel
        thickness = random.randint(1, max_thickness)  # Uniform thin scratch
        for j in range(-thickness, thickness + 1):
            rr_offset = continuous_rr + j
            cc_offset = continuous_cc
            valid_indices = (0 <= rr_offset) & (rr_offset < h) & (0 <= cc_offset) & (cc_offset < w)

            # Apply random normal distribution for each pixel in the scratch
            if random.random() < 0.5:  # scratch is dark
                mean_intensity = random.randint(20, 50)
                std_dev_intensity = random.randint(5, 10)
            else:  # scratch is bright
                mean_intensity = random.randint(130, 150)
                std_dev_intensity = random.randint(5, 10)
            noise = np.random.normal(mean_intensity, std_dev_intensity, size=(len(rr_offset[valid_indices]), 3))
            noise = np.clip(noise, 0, 255).astype(np.uint8)

            img[rr_offset[valid_indices], cc_offset[valid_indices]] = np.clip(noise, 0, 255).astype(np.uint8)

        # Add bright halo effect with a probability
        if random.random() < 0.5:  # 50% chance of having a halo
            halo_layer = np.zeros_like(img, dtype=np.float32)
            halo_width = random.randint(1, 5)  # Uniform halo width
            for j in range(thickness + 1, thickness + halo_width + 1):
                rr_offset = continuous_rr + j
                cc_offset = continuous_cc
                valid_indices = (0 <= rr_offset) & (rr_offset < h) & (0 <= cc_offset) & (cc_offset < w)
                halo_layer[rr_offset[valid_indices], cc_offset[valid_indices]] = random.randint(180, 255)  # Bright halo

            # Apply Gaussian blur to smooth the halo
            halo_layer = gaussian(halo_layer, sigma=1.5, channel_axis=-1, truncate=4.0)
            img = cv2.addWeighted(img.astype(np.float32), 1.0, halo_layer, 0.4, 0).astype(np.uint8)

        mask[rr_offset[valid_indices], cc_offset[valid_indices]] = 255

        # Define bounding box around the scratch
        bounding_box = [min(continuous_cc), min(continuous_rr), max(continuous_cc), max(continuous_rr)]

    elif defect_type == "golden_bumps":
        # Define parameters for the random crater
        center_x, center_y = random.randint(50, w - 50), random.randint(50, h - 50)  # Random center
        radius = random.randint(3, 20)  # Base radius

        # Define the normal distribution parameters for color (mean and std)
        mean_color = random.randint(10, 30)  # Mean color intensity
        std_color = random.randint(5, 12)  # Color intensity variability

        # Generate random RGB values using normal distribution for each pixel
        # Create a random color map for the entire image using the normal distribution
        color_map_r = np.clip(np.random.normal(mean_color, std_color, size=(h, w)), 0, 255)
        color_map_g = np.clip(np.random.normal(mean_color, std_color, size=(h, w)), 0, 255)
        color_map_b = np.clip(np.random.normal(mean_color, std_color, size=(h, w)), 0, 255)

        # Stack the 3 channels to create an RGB image
        color_map = np.stack([color_map_r, color_map_g, color_map_b], axis=-1).astype(np.uint8)

        # Create a mask for the crater
        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)

        # Generate irregular edges for the crater
        angles = np.linspace(0, 2 * np.pi, num=100)
        x_coords = []
        y_coords = []

        for angle in angles:
            rand_radius = radius + random.randint(-2, 2)  # Slightly irregular edges
            x = int(center_x + rand_radius * np.cos(angle))
            y = int(center_y + rand_radius * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                x_coords.append(x)
                y_coords.append(y)

        # Draw the irregular crater using a filled polygon
        points = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # Create an edge mask by subtracting an eroded version of the mask
        eroded_mask = cv2.erode(mask, np.ones((4, 4), np.uint8), iterations=1)  # Inner mask
        edge_mask = cv2.subtract(mask, eroded_mask)  # Edge area only

        # Apply Gaussian blur to smooth the edges
        smoothed_edge_mask = cv2.GaussianBlur(edge_mask, (15, 15), sigmaX=5, sigmaY=5)

        # Blend only the edges of the crater with the original image
        for c in range(3):  # Iterate over color channels
            img[:, :, c] = np.where(
                smoothed_edge_mask > 0,
                (img[:, :, c].astype(np.float32) * (1 - smoothed_edge_mask / 255 * 0.6)).astype(np.uint8),
                img[:, :, c]
            )

        # Now, apply the generated random color map to the image for the crater region
        img[eroded_mask > 0] = color_map[eroded_mask > 0]

        cv2.fillPoly(mask, [points], 255)

        # Define bounding box for the golden bump
        bounding_box = [
            max(0, center_x - radius),
            max(0, center_y - radius),
            min(w, center_x + radius),
            min(h, center_y + radius)
        ]

    elif defect_type == "ring":
        # Define parameters for the ellipse
        center_x, center_y = random.randint(50, w - 50), random.randint(50, h - 50)  # Random center
        axes = (random.randint(5, 15), random.randint(5, 15))  # Major and minor axes lengths
        angle = random.randint(0, 360)  # Rotation angle
        color_black = (random.randint(0, 20), random.randint(0, 20), random.randint(0, 20))  # Gray-black color
        color_gray = (random.randint(40, 80), random.randint(40, 80), random.randint(40, 80))
        color = random.choice([color_black, color_gray])  # Randomly choose black or gray color
        thickness = random.randint(1, 1)  # Thickness of the ellipse boundary

        # Draw the ellipse on the image
        cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, color, thickness)

        # Update the mask for the ring area
        cv2.ellipse(mask, (center_x, center_y), axes, angle, 0, 360, 255, -1)

        # Define bounding box for the defect
        bounding_box = [
            max(0, center_x - axes[0]),
            max(0, center_y - axes[1]),
            min(w, center_x + axes[0]),
            min(h, center_y + axes[1])
        ]

    elif defect_type == "gate_pollution":
        # Parameters for the defect
        start_x, start_y = random.randint(50, w - 50), random.randint(50, h - 50)  # Random starting point
        length = random.randint(4, 100)  # Total length of the hair
        hair_thickness = random.randint(1, 2)  # Thin hairline

        # Generate the hair path
        points = [(start_x, start_y)]
        angle = random.uniform(0, 2 * np.pi)  # Random initial direction
        curve_start = int(length * 0.7)  # Start curving at the last 30% of the hair

        for i in range(length):
            if i < curve_start:
                # Straight section: minimal angle change
                angle += random.uniform(-np.pi / 16, np.pi / 16)
            else:
                # Curved section: gradual spiral
                angle += random.uniform(-np.pi / 8, np.pi / 8)

            # Generate the next point
            x = int(points[-1][0] + random.randint(3, 5) * np.cos(angle))
            y = int(points[-1][1] + random.randint(3, 5) * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:  # Ensure the point is within image bounds
                points.append((x, y))

        # Draw the hair defect with variable color intensity
        mean_intensity = random.randint(0, 30)  # Mean color intensity
        std_dev_intensity = random.randint(5, 12)  # Color intensity variability
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            color = tuple(
                map(int, np.clip(np.random.normal(mean_intensity, std_dev_intensity, size=3), 0, 255).astype(np.uint8)))
            cv2.line(img, pt1, pt2, color, hair_thickness)
            cv2.line(mask, pt1, pt2, 255, hair_thickness)  # Draw the hairline on the mask

        # Define bounding box for the gate-pollution defect
        all_x = [p[0] for p in points]
        all_y = [p[1] for p in points]
        bounding_box = [
            max(0, min(all_x) - hair_thickness),
            max(0, min(all_y) - hair_thickness),
            min(w, max(all_x) + hair_thickness),
            min(h, max(all_y) + hair_thickness)
        ]

    elif defect_type == "contamination":
        # Number of spray areas (1 to 3)
        num_areas = random.randint(1, 4)

        # Base center for overlapping areas
        base_center_x, base_center_y = random.randint(50, w - 50), random.randint(50, h - 50)

        color_black = random.randint(0, 30)
        color_gray = random.randint(40, 80)

        mean_color = random.choice([color_black, color_gray])  # Randomly choose black or gray color
        std_color = random.randint(5, 12)  # Color intensity variability

        color_map_r = np.clip(np.random.normal(mean_color, std_color, size=(h, w)), 0, 255)
        color_map_g = np.clip(np.random.normal(mean_color, std_color, size=(h, w)), 0, 255)
        color_map_b = np.clip(np.random.normal(mean_color, std_color, size=(h, w)), 0, 255)

        # Stack the 3 channels to create an RGB image
        color_map = np.stack([color_map_r, color_map_g, color_map_b], axis=-1).astype(np.uint8)

        cur_base_center_x, cur_base_center_y = base_center_x, base_center_y
        for _ in range(num_areas):
            # Parameters for each spray area
            offset_x = random.randint(-15, 15)  # Offset to make them close to each other
            offset_y = random.randint(-15, 15)
            center_x = max(0, min(w - 1, cur_base_center_x + offset_x))
            center_y = max(0, min(h - 1, cur_base_center_y + offset_y))

            spray_radius = random.randint(5, 25)  # Radius of the spray area
            # Determine spray density as a percentage of the circle's area
            spray_percentage = random.uniform(0.5, 0.9)  # Percentage range (1% to 5% of the circle's area)

            spray_area = np.pi * spray_radius ** 2
            spray_density = int(spray_percentage * spray_area)  # Number of spray dots based on circle area

            cv2.circle(mask, (center_x, center_y), spray_radius, 255, -1)  # Fill contamination in the mask

            for _ in range(spray_density):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, spray_radius)
                dot_x = int(center_x + radius * np.cos(angle))
                dot_y = int(center_y + radius * np.sin(angle))
                if 0 <= dot_x < w and 0 <= dot_y < h:  # Ensure the dot is within image bounds
                    img[dot_y, dot_x] = color_map[dot_y, dot_x]

            cur_base_center_x, cur_base_center_y = center_x, center_y

        # Define bounding box for the entire contamination defect
        bounding_box = [
            max(0, base_center_x - 50),
            max(0, base_center_y - 50),
            min(w, base_center_x + 50),
            min(h, base_center_y + 50)
        ]

    elif defect_type == "holy_white":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ksize = random.choice([3, 5, 7])
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Apply edge detection (Canny)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=70)

        # Create a blank image for the glow effect
        glow = np.zeros_like(image)

        # Find the coordinates of the edge points
        edge_points = np.column_stack(np.where(edges > 0))

        # Focus on a random edge point
        glow_point = edge_points[np.random.choice(edge_points.shape[0])]  # Random edge point

        # Randomize the radius size for the glowing effect
        radius = np.random.randint(3, 10)  # Random radius between min and max values

        # Add a bright glowing effect at this point
        center = tuple(glow_point[::-1])  # Convert (y, x) to (x, y) format
        cv2.circle(glow, center, radius=radius, color=(255, 255, 255), thickness=-1)  # Bright white circle

        # Apply Gaussian blur to make the glow smooth
        glow = cv2.GaussianBlur(glow, (15, 15), sigmaX=5, sigmaY=5)

        # Increase brightness by intensity factor
        intensity_factor = random.uniform(1.0, 2.0)
        glow = np.clip(glow * intensity_factor, 0, 255).astype(np.uint8)  # Make it brighter

        # Blend the glow with the original image
        glowing_image = cv2.addWeighted(image, 1.0, glow, 0.8, 0)

        # Update the mask for the glow area
        cv2.circle(mask, center, radius, 255, -1)  # Mark the glowing area in the mask

        img = glowing_image

        # Define bounding box for the entire contamination defect
        bounding_box = [
            max(0, glow_point[1] - radius),
            max(0, glow_point[0] - radius),
            min(w, glow_point[1] + radius),
            min(h, glow_point[0] + radius)
        ]

    # add fixed size to the bbox to ensure the whole abject is inside the bbox
    # the bounding box format in this position:
    # bounding_box = [max_x, max_y, min_x, min_y]
    if bounding_box:
        bounding_box[0] = max(0, bounding_box[0] - 10)
        bounding_box[1] = max(0, bounding_box[1] - 10)
        bounding_box[2] = min(w, bounding_box[2] + 10)
        bounding_box[3] = min(h, bounding_box[3] + 10)

    return img, mask, bounding_box

def convert_mask_to_label(mask):
    """
    Convert mask (image_size, image_size) to label in txt file
    each line in the txt file is in the format:
    class_number x0 y0 x1 y1 ... xn yn
    where x0, y0, x1, y1, ... xn, yn are the points of the segmentation of the object
    :param mask:
    :return: list of touples, each touple is (class_number, [list of points])
    list of points format: [x0, y0, x1, y1, ... xn, yn] and they ara normalized by the image size
    """

    # get the unique values in the mask
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values != 0]

    # list of touples, each touple is (class_number, [list of points])
    labels = []

    for value in unique_values:
        # get the points of the object
        points = np.argwhere(mask == value)

        # normalize the points by h, w
        h, w = mask.shape
        points = points / [h, w]

        points = points.flatten().tolist()

        labels.append((value, points))

    return labels


def create_pairs(clean_images, task, num_images=500):
    """Create synthetic dataset pairs."""
    os.makedirs(f"{OUTPUT_DIR}/images/reference", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/inspected", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/combined", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/annotated", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/annotated_mask", exist_ok=True)

    os.makedirs(f"{OUTPUT_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/images/test", exist_ok=True)

    os.makedirs(f"{OUTPUT_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/test", exist_ok=True)

    os.makedirs(f"{OUTPUT_DIR}/masks/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/masks/val", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/masks/test", exist_ok=True)

    for i in tqdm(range(num_images)):
        clean_image = random.choice(clean_images)
        clean_image = cv2.resize(clean_image, IMAGE_SIZE)

        # Augment reference image
        clean_image = augment_image(clean_image)

        # Decide the number of defects
        defect_count = \
            random.choices(list(defect_probabilities.keys()), weights=list(defect_probabilities.values()), k=1)[0]

        # Generate inspected image
        inspected_image = clean_image.copy()

        # augment inspected image
        inspected_image = augment_image(inspected_image, image_type='inspected')

        bounding_boxes = []
        mask = np.zeros_like(inspected_image[:, :, 0])

        for _ in range(defect_count):
            cls_num, defect_type = random_defect_type()
            inspected_image, defect_mask, bbox = inject_defect(inspected_image, defect_type)
            if bbox:
                bounding_boxes.append([cls_num,
                                       (bbox[0] + bbox[2]) / 2 / IMAGE_SIZE[1],
                                       (bbox[1] + bbox[3]) / 2 / IMAGE_SIZE[0],
                                       abs((bbox[2] - bbox[0]) / IMAGE_SIZE[1]),
                                       abs((bbox[3] - bbox[1]) / IMAGE_SIZE[0])
                                       ])
                mask[defect_mask == 255] = cls_num

        # shrink and resize
        clean_image = cv2.resize(clean_image, (int(IMAGE_SIZE[1] * 0.75), int(IMAGE_SIZE[0] * 0.75)))
        inspected_image = cv2.resize(inspected_image, (int(IMAGE_SIZE[1] * 0.75), int(IMAGE_SIZE[0] * 0.75)))

        clean_image = cv2.resize(clean_image, IMAGE_SIZE)
        inspected_image = cv2.resize(inspected_image, IMAGE_SIZE)

        # Save reference image
        ref_path = f"{OUTPUT_DIR}/images/reference/ref_{i:04d}.png"
        cv2.imwrite(ref_path, clean_image)

        # Save inspected image
        insp_path = f"{OUTPUT_DIR}/images/inspected/insp_{i:04d}.png"
        cv2.imwrite(insp_path, inspected_image)

        # train or val or test
        train_or_val_or_test = 'train' if random.random() < 0.7 else 'val' if random.random() < 0.5 else 'test'

        # Save combined image
        # switch the images
        if random.random() < 0.5:
            clean_image, inspected_image = inspected_image, clean_image
        combined_image = preprocess_images(Image.fromarray(clean_image), Image.fromarray(inspected_image))
        combined_path = f"{OUTPUT_DIR}/images/{train_or_val_or_test}/image_{i}.png"
        cv2.imwrite(combined_path, combined_image)

        label_path = f"{OUTPUT_DIR}/labels/{train_or_val_or_test}/image_{i}.txt"
        if task == "defect_detection":
            with open(label_path, "w") as f:
                for bbox in bounding_boxes:
                    f.write(" ".join(map(str, bbox)) + "\n")

        elif task == "segmentation":
            labels_from_mask = convert_mask_to_label(mask)
            with open(label_path, "w") as f:
                for label, points in labels_from_mask:
                    f.write(f"{label} {' '.join(map(str, points))}\n")

            mask_path = f"{OUTPUT_DIR}/masks/{train_or_val_or_test}/mask_{i}.png"
            cv2.imwrite(mask_path, mask)

        # save annotated mask<255 and make it 3 channels
        annotated_mask = mask.copy()
        annotated_mask[annotated_mask > 0] = 255
        annotated_mask = np.stack([annotated_mask, annotated_mask, annotated_mask], axis=-1)

        # Create a copy of the inspected image for annotation
        annotated_image = inspected_image.copy()

        # Draw bounding boxes and labels on the annotated image
        for bbox in bounding_boxes:
            cls_num, center_x, center_y, width, height = bbox
            x_min = int((center_x - width / 2) * IMAGE_SIZE[1])
            y_min = int((center_y - height / 2) * IMAGE_SIZE[0])
            x_max = int((center_x + width / 2) * IMAGE_SIZE[1])
            y_max = int((center_y + height / 2) * IMAGE_SIZE[0])

            # Draw rectangle (bounding box)
            cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.rectangle(annotated_mask, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Add label text above the bounding box
            label = f"Defect {cls_num}"
            cv2.putText(
                annotated_image,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            cv2.putText(
                annotated_mask,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        # Save annotated image
        annotated_path = f"{OUTPUT_DIR}/images/annotated/annotated_{i:04d}.png"
        cv2.imwrite(annotated_path, annotated_image)

        # Save annotated mask
        annotated_mask_path = f"{OUTPUT_DIR}/images/annotated_mask/annotated_mask_{i:04d}.png"
        cv2.imwrite(annotated_mask_path, annotated_mask)


if __name__ == "__main__":
    # task
    task = "segmentation"  # detection or segmentation

    # Output directory
    OUTPUT_DIR = f"synthetic_dataset_{task}"

    # Load clean chip images
    clean_image_paths = ["data/defective_examples/case1_reference_image.tif",
                         "data/defective_examples/case2_reference_image.tif",
                         "data/non_defective_examples/case3_reference_image.tif"]
    clean_images = [cv2.imread(p) for p in clean_image_paths]

    # Create synthetic dataset
    create_pairs(clean_images, task, num_images=NUM_IMAGES)
