import cv2
import numpy as np
import random
from skimage.metrics import structural_similarity as ssim


def align_images_shift(img1, img2, roi_x, roi_y, roi_size):
    """Align img2 to img1 by finding the optimal translation (up, down, left, right) to match the images in a given region of interest (ROI)."""
    # Step 1: Define the region of interest (ROI) around the given coordinates and size
    h, w = img1.shape[:2]
    x_min = max(0, roi_x - roi_size)
    x_max = min(w, roi_x + roi_size)
    y_min = max(0, roi_y - roi_size)
    y_max = min(h, roi_y + roi_size)

    # Step 2: Extract ROI from both images
    gray1_roi = cv2.cvtColor(img1[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
    gray2_roi = cv2.cvtColor(img2[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)

    # Step 3: Detect ORB features in the cropped region of interest
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1_roi, None)
    kp2, des2 = orb.detectAndCompute(gray2_roi, None)

    # Step 4: Match features using Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Step 5: Sort matches by distance (good matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Step 6: Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Step 7: Find the translation vector (shift) using the matched keypoints
    translation = np.mean(dst_pts - src_pts, axis=0)
    shift_x, shift_y = translation[0][0], translation[0][1]

    # Step 8: Apply the translation to shift the inspected image
    rows, cols = img2.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    aligned_img2 = cv2.warpAffine(img2, M, (cols, rows))

    return aligned_img2, matches, x_min, y_min, x_max, y_max


def evaluate_alignment(reference_img, inspected_img, search_area_size=100, num_tries=10):
    """Try multiple alignments and choose the one with the smallest difference using SSIM."""
    best_alignment = None
    best_score = -1  # SSIM score ranges from -1 to 1, so we start with a low value

    for _ in range(num_tries):
        # Randomly select a center for the ROI within the image bounds
        h, w = reference_img.shape[:2]
        roi_x = random.randint(0, w)
        roi_y = random.randint(0, h)

        # Align the images using this region of interest
        aligned_img, matches, x_min, y_min, x_max, y_max = align_images_shift(reference_img, inspected_img, roi_x,
                                                                              roi_y, search_area_size)

        # Crop both the reference and aligned images to the overlapping region
        cropped_ref = reference_img[y_min:y_max, x_min:x_max]
        cropped_aligned = aligned_img[y_min:y_max, x_min:x_max]

        # Calculate similarity using SSIM (Structural Similarity Index)
        score, _ = ssim(cropped_ref, cropped_aligned, full=True, win_size=3)

        # If this alignment has the best SSIM score, update best alignment
        if score > best_score:
            best_score = score
            best_alignment = aligned_img

    return best_alignment, best_score


def detect_defects(reference_img, inspected_img, threshold=50):
    """Detect defects by comparing the reference image and inspected image."""
    # Convert images to grayscale
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    insp_gray = cv2.cvtColor(inspected_img, cv2.COLOR_BGR2GRAY)

    # Find absolute difference between the images
    diff = cv2.absdiff(ref_gray, insp_gray)

    # Threshold the difference to detect defects
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return mask


def display_images_with_mask(reference_img, inspected_img, mask_img):
    """Display the reference, inspected images along with the defect mask."""
    # Resize images for display purposes
    reference_img_resized = cv2.resize(reference_img, (0, 0), fx=0.75, fy=0.75)
    inspected_img_resized = cv2.resize(inspected_img, (0, 0), fx=0.75, fy=0.75)
    mask_img_resized = cv2.resize(mask_img, (0, 0), fx=0.75, fy=0.75)

    # Convert the mask to a 3-channel image (to match reference and inspected images)
    mask_img_resized_3d = cv2.cvtColor(mask_img_resized, cv2.COLOR_GRAY2BGR)

    # Stack the images side by side
    stacked_img = np.hstack((reference_img_resized, inspected_img_resized, mask_img_resized_3d))

    # Display the images
    cv2.imshow("Reference | Inspected | Defect Mask", stacked_img)


if __name__ == "__main__":
    # Load the reference and inspected images (replace with your own paths)
    reference_apth = "data/defective_examples/case2_reference_image.tif"
    inspected_apth = "data/defective_examples/case2_inspected_image.tif"

    # Load images
    reference_img = cv2.imread(reference_apth)
    inspected_img = cv2.imread(inspected_apth)

    # Evaluate multiple alignments and choose the one with the best SSIM score
    best_aligned_img, best_score = evaluate_alignment(reference_img, inspected_img, search_area_size=100, num_tries=10)

    print(f"Best SSIM Score: {best_score}")

    # Detect defects in the best alignment
    defect_mask = detect_defects(reference_img, best_aligned_img, threshold=50)

    # Display the results with the defect mask
    display_images_with_mask(reference_img, best_aligned_img, defect_mask)

    # Wait until the user closes the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
