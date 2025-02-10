import cv2
import matplotlib.pyplot as plt
import numpy as np


# Function to load and preprocess images
def load_and_preprocess_images(img1, img2):
    # Load images if its paths:
    if isinstance(img1, str):
        img1 = cv2.imread(img1)
    if isinstance(img2, str):
        img2 = cv2.imread(img2)

    # Resize image2 to match the dimensions of image1
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Resize images to the same size
    img1 = cv2.resize(img1, (640, 640))
    img2 = cv2.resize(img2, (640, 640))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    # gray1 = cv2.equalizeHist(gray1)
    # gray2 = cv2.equalizeHist(gray2)

    # Apply Gaussian blur
    # gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    # gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    return gray1, gray2, img1, img2


# Function to align two images using ORB and Affine transformation
def align_images(img1, img2, distance_threshold=40, transformation_type='affine'):
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # Detect keypoints and descriptors in both images
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Use BFMatcher to find the best matches between the two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort the matches based on distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter out matches based on the distance threshold
    good_matches = []
    for match in matches:
        kp1_coords = kp1[match.queryIdx].pt
        kp2_coords = kp2[match.trainIdx].pt
        distance = np.linalg.norm(np.array(kp1_coords) - np.array(kp2_coords))  # Euclidean distance
        if distance < distance_threshold:
            good_matches.append(match)

    # Extract the filtered matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Apply the selected transformation
    if transformation_type == 'affine':
        # Find the Affine transformation matrix using the good matches
        M, mask = cv2.estimateAffine2D(pts2, pts1)  # Using Affine transformation
    elif transformation_type == 'euclidean':
        # Find the Euclidean transformation matrix using the good matches
        M, mask = cv2.estimateAffine2D(pts2, pts1, method=cv2.RANSAC,
                                       ransacReprojThreshold=3.0)  # Euclidean (translation + rotation)
    else:
        raise ValueError("Invalid transformation_type. Choose either 'affine' or 'euclidean'.")

    # Warp img2 to align with img1 using the transformation matrix
    height, width = img1.shape
    aligned_img2 = cv2.warpAffine(img2, M, (width, height))

    # Draw the good matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return M, good_matches, match_img


def get_bounding_box_for_crop(img1, img2, M):
    # Convert images to grayscale
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # create a same size image with one's
    mask = np.ones_like(img1, dtype=np.uint8) * 255

    # apply the transformation to the mask
    mask = cv2.warpAffine(mask, M, (img1.shape[1], img1.shape[0]))

    # Find contours of the combined mask (non-zero regions)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the combined mask (non-zero region)
    x, y, w, h = cv2.boundingRect(contours[0])

    return x, y, w, h


def display_images_side_by_side(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title('Reference Image')
    ax[0].axis('off')
    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title('Aligned Image')
    ax[1].axis('off')
    plt.show()


def display_matches(match_img):
    plt.figure(figsize=(12, 6))
    plt.imshow(match_img)
    plt.title("Keypoint Matches")
    plt.axis('off')
    plt.show()


def find_best_aligment(img1, img2):
    transformation_types = ['affine', 'euclidean']
    kp_distance_thresholds = [40, 50, 60, 70, 80, 90, 100]
    best_similarity = 0
    best_M = None
    best_cropped_images = None

    gray1, gray2, img1, img2 = load_and_preprocess_images(img1, img2)

    for transformation_type in transformation_types:
        for distance_threshold in kp_distance_thresholds:
            M, _, _ = align_images(gray1, gray2, distance_threshold, transformation_type)
            aligned_gray2 = cv2.warpAffine(gray2, M, gray1.shape[:2])
            x, y, w, h = get_bounding_box_for_crop(gray1, aligned_gray2, M)
            little_more = 10
            cropped_gray1 = gray1[y + little_more:y + h - little_more, x + little_more:x + w - little_more]
            cropped_aligned_gray2 = aligned_gray2[y + little_more:y + h - little_more,
                                    x + little_more:x + w - little_more]

            # Calculate the similarity between the cropped images
            similarity = cv2.matchTemplate(cropped_gray1, cropped_aligned_gray2, cv2.TM_CCOEFF_NORMED)[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_M = M

                aligned_img2 = cv2.warpAffine(img2, M, img1.shape[:2])
                cropped_img1 = img1[y + little_more:y + h - little_more, x + little_more:x + w - little_more]
                cropped_aligned_img2 = aligned_img2[y + little_more:y + h - little_more,
                                       x + little_more:x + w - little_more]
                best_cropped_images = (cropped_img1, cropped_aligned_img2)

    print(f"Best similarity: {best_similarity}")
    return best_cropped_images


if __name__ == "__main__":
    # Example usage
    # image1_path = "data/defective_examples/case2_reference_image.tif"
    # image2_path = "data/defective_examples/case2_inspected_image.tif"

    # image1_path = "data/defective_examples/case1_reference_image.tif"
    # image2_path = "data/defective_examples/case1_inspected_image.tif"

    image1_path = "synthetic_dataset/insp_0001.png"
    image2_path = "synthetic_dataset/ref_0001.png"

    # # Load and preprocess images
    # gray1, gray2, img1, img2 = load_and_preprocess_images(image1_path, image2_path)
    #
    # # Align the images using Euclidean transformation
    # transformation_type = 'affine'  # Choose either 'affine' or 'euclidean'
    # M, good_matches, match_img = align_images(gray1, gray2, transformation_type=transformation_type)
    #
    # # Apply the Euclidean transformation to the original image
    # aligned_img2 = cv2.warpAffine(img2, M, img1.shape[:2])
    #
    # # Get the bounding box for cropping the aligned image
    # x, y, w, h = get_bounding_box_for_crop(img1, aligned_img2, M)
    # little_more = 10
    #
    # # Crop the both images
    # cropped_img1 = img1[y+little_more:y+h-little_more, x+little_more:x+w-little_more]
    # cropped_aligned_img2 = aligned_img2[y+little_more:y+h-little_more, x+little_more:x+w-little_more]
    #
    # # Display gray images side by side
    # aligned_gray2 = cv2.warpAffine(gray2, M, gray1.shape[:2])
    # display_images_side_by_side(gray1, aligned_gray2)
    #
    # # Display the keypoint matches
    # display_matches(match_img)
    #
    # # Display images side by side
    # display_images_side_by_side(img1, aligned_img2)
    #
    # # Display the cropped images side by side
    # display_images_side_by_side(cropped_img1, cropped_aligned_img2)

    cropped_img1, cropped_aligned_img2 = find_best_aligment(image1_path, image2_path)
    # Display the cropped images side by side
    display_images_side_by_side(cropped_img1, cropped_aligned_img2)
