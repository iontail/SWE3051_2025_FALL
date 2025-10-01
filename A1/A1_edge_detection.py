import cv2
import numpy as np

from A1_image_filtering import cross_correlation_2d, cross_correlation_1d, get_gaussian_filter_2d, get_gaussian_filter_1d

def compute_image_gradient(img: np.ndarray):





if __name__=="__main__":
    lenna = cv2.imread('./A1_Images/lenna.png', cv2.IMREAD_GRAYSCALE)
    shapes = cv2.imread('./A1_Images/shapes.png', cv2.IMREAD_GRAYSCALE)
    lenna = np.asarray(lenna).astype(np.float32)
    shapes = np.asarray(shapes).astype(np.float32)

    filter = get_gaussian_filter_2d(7, 1.5)
    filtered_img = cross_correlation_2d(lenna, filter)

    cv2.imshow("Gaussian Filtered lenna.png", np.clip(filtered_img, 0, 255).astype(np.uint8))
    cv2.waitKey(0)