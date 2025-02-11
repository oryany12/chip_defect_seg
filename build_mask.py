import cv2
import numpy as np
from alignment import load_and_preprocess_images, find_best_aligment


# Function to subtract images and get the difference
def subtract_images(img1, img2, threshold):
    diff = cv2.absdiff(img1, img2)
    _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return diff_mask


# Function to detect defects based on the difference and apply masking
def detect_defects(img1, img2, threshold_value=50, blur_radius=5, canny_low=100, canny_high=200, morph_kernel_size=5):
    gray1, gray2, _, _ = load_and_preprocess_images(img1, img2)

    # Subtract images and get mask
    diff_mask = subtract_images(gray1, gray2, threshold_value)

    # Ensure blur_radius is a valid odd number
    if blur_radius % 2 == 0:
        blur_radius += 1  # Make it odd if it's even

    # Apply Gaussian blur to smooth the mask
    blurred_mask = cv2.GaussianBlur(diff_mask, (blur_radius, blur_radius), 0)

    # Histogram equalization to enhance contrast
    enhanced_mask = cv2.equalizeHist(blurred_mask)

    # Threshold to isolate bright white spots (defects)
    _, defect_mask = cv2.threshold(enhanced_mask, 200, 255, cv2.THRESH_BINARY)

    # Apply edge detection (Canny)
    edges = cv2.Canny(defect_mask, canny_low, canny_high)

    # Morphological operations to clean the mask
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    cleaned_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask, defect_mask


# Function to display images and interactively adjust parameters
def interactive_defect_detection(img1, img2):
    def nothing(x):
        pass

    # Create a window for interactive adjustments
    cv2.namedWindow("Defect Detection")

    # Initialize default values for sliders
    default_threshold = 50
    default_blur_radius = 5
    default_canny_low = 100
    default_canny_high = 200
    default_morph_kernel_size = 5

    # Create trackbars (sliders) for threshold, blur radius, and other parameters
    cv2.createTrackbar("Threshold", "Defect Detection", default_threshold, 255, nothing)
    cv2.createTrackbar("Blur Radius", "Defect Detection", default_blur_radius, 20, nothing)
    cv2.createTrackbar("Canny Low", "Defect Detection", default_canny_low, 255, nothing)
    cv2.createTrackbar("Canny High", "Defect Detection", default_canny_high, 255, nothing)
    cv2.createTrackbar("Morph Kernel Size", "Defect Detection", default_morph_kernel_size, 20, nothing)

    while True:
        # Get the current slider values
        threshold_value = cv2.getTrackbarPos("Threshold", "Defect Detection")
        blur_radius = cv2.getTrackbarPos("Blur Radius", "Defect Detection")
        canny_low = cv2.getTrackbarPos("Canny Low", "Defect Detection")
        canny_high = cv2.getTrackbarPos("Canny High", "Defect Detection")
        morph_kernel_size = cv2.getTrackbarPos("Morph Kernel Size", "Defect Detection")

        # Get the defect mask with the current parameter values
        cleaned_mask, defect_mask = detect_defects(
            img1, img2, threshold_value, blur_radius, canny_low, canny_high, morph_kernel_size
        )

        # Stack the images for display
        blurred_mask_3ch = cv2.merge((cleaned_mask, cleaned_mask, cleaned_mask))
        defect_mask_3ch = cv2.merge((defect_mask, defect_mask, defect_mask))
        img1_for_display = cv2.resize(img1, blurred_mask_3ch.shape[:2])
        img2_for_display = cv2.resize(img2, blurred_mask_3ch.shape[:2])
        combined = cv2.hconcat([img1_for_display, blurred_mask_3ch, defect_mask_3ch, img2_for_display])

        # Show the combined images and defect mask
        cv2.imshow("Defect Detection", combined)

        # Wait for a key press to exit the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit the loop
            break

    # Cleanup
    cv2.destroyAllWindows()


# Example usage with image paths
if __name__ == "__main__":
    image1_path = "data/defective_examples/case2_reference_image.tif"
    image2_path = "data/defective_examples/case2_inspected_image.tif"

    image1_path = "synthetic_dataset/insp_0019.png"
    image2_path = "synthetic_dataset/ref_0019.png"

    gray1, gray2, img1, img2 = load_and_preprocess_images(image1_path, image2_path)

    cropped_img1, cropped_aligned_img2 = find_best_aligment(img1, img2)

    interactive_defect_detection(cropped_img1, cropped_aligned_img2)
