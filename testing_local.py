import cv2
import numpy as np
import matplotlib.pyplot as plt


def align_images_using_orb(img1, img2):
    """Align img2 to img1 using ORB feature matching and homography, keeping only the overlapping area."""

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=5000)

    # Detect keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between the two images
    matches = bf.match(des1, des2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw the first 10 matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract the matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate the homography matrix using RANSAC
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # Warp img2 to align with img1
    aligned_img2 = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    # Compute the overlap area
    mask1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2GRAY) > 0
    overlap_mask = np.logical_and(mask1, mask2)

    # Find bounding box of the overlapping area
    y_indices, x_indices = np.where(overlap_mask)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    min_y, max_y = np.min(y_indices), np.max(y_indices)

    # Crop both images to the overlapping region
    cropped_img1 = img1[min_y:max_y, min_x:max_x]
    cropped_img2 = aligned_img2[min_y:max_y, min_x:max_x]

    return cropped_img1, cropped_img2, img_matches


# Function to display images side by side
def display_images(reference_img, inspected_img, img_matches):
    # Resize images for display purposes
    reference_img_resized = cv2.resize(reference_img, (0, 0), fx=0.75, fy=0.75)
    inspected_img_resized = cv2.resize(inspected_img, (0, 0), fx=0.75, fy=0.75)

    # Stack the images side by side
    stacked_img = np.hstack((reference_img_resized, inspected_img_resized))

    # Display the images and matches
    cv2.imshow("Aligned Reference | Aligned Inspected", stacked_img)
    cv2.imshow("Feature Matches", img_matches)


# Load images (replace with your file paths)
reference_apth = "data/defective_examples/case2_reference_image.tif"
inspected_apth = "data/defective_examples/case2_inspected_image.tif"

reference_img = cv2.imread(reference_apth)
inspected_img = cv2.imread(inspected_apth)

# Align the images using ORB
aligned_reference, aligned_inspected, img_matches = align_images_using_orb(reference_img, inspected_img)

# Display the aligned images and feature matches
display_images(aligned_reference, aligned_inspected, img_matches)

# Wait until the user closes the window
cv2.waitKey(0)
cv2.destroyAllWindows()
