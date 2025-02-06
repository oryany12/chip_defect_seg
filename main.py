import cv2
import numpy as np


# Your alignment function
def align_images(img1, img2):
    """Align img2 to img1 using feature matching and homography, keeping only the overlapping area."""

    # Step 1: Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply contrast normalization to address brightness differences (optional but useful)
    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)

    # Step 3: Detect features using ORB (you can also use SIFT or SURF if needed)
    orb = cv2.ORB_create(10000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Step 4: Match features using Brute Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Step 5: Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Step 6: Compute homography matrix using RANSAC (robust to outliers)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Step 7: Apply perspective transformation (alignment)
    aligned_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    # Step 8: Compute the overlapping region (non-black area)
    mask1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY) > 0
    overlap_mask = np.logical_and(mask1, mask2)

    # Step 9: Find bounding box of the overlapping area
    y_indices, x_indices = np.where(overlap_mask)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)

    # Step 10: Crop both images to the overlapping region
    cropped_img1 = img1[min_y:max_y, min_x:max_x]
    cropped_img2 = aligned_img2[min_y:max_y, min_x:max_x]

    return cropped_img1, cropped_img2


# Function to detect defects using image difference
def detect_defects(reference_img, inspected_img, threshold=50):
    # Convert to grayscale
    ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    insp_gray = cv2.cvtColor(inspected_img, cv2.COLOR_BGR2GRAY)

    # Find absolute difference between the images
    diff = cv2.absdiff(ref_gray, insp_gray)

    # Threshold the difference to detect defects
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return mask


# Function to display images and the defect mask
def display_images_with_mask(reference_img, inspected_img, mask_img):
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


# Function to update and control defect detection
def update(val):
    threshold = cv2.getTrackbarPos('Threshold', 'Controls')

    # Detect defects using the current threshold
    mask = detect_defects(aligned_img_reference, aligned_img_inspected, threshold)

    # Display the images and mask
    display_images_with_mask(aligned_img_reference, aligned_img_inspected, mask)


reference_apth = "data/defective_examples/case2_reference_image.tif"
inspected_apth = "data/defective_examples/case2_inspected_image.tif"

# Load images
reference_img = cv2.imread(reference_apth)
inspected_img = cv2.imread(inspected_apth)

# Align the images
aligned_img_reference, aligned_img_inspected = align_images(reference_img, inspected_img)

# Create a window for interactive controls
cv2.namedWindow('Controls')
cv2.createTrackbar('Threshold', 'Controls', 50, 255, update)  # Threshold for defect detection

# Initial call to update the display
update(0)

# Wait until the user closes the window
cv2.waitKey(0)
cv2.destroyAllWindows()
