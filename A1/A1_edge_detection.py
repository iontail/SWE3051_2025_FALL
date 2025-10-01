import cv2
import numpy as np

from A1_image_filtering import cross_correlation_2d, cross_correlation_1d, get_gaussian_filter_2d, get_gaussian_filter_1d

def compute_image_gradient(img: np.ndarray):
    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    grad_x = cross_correlation_2d(img, sobel_x)
    grad_y = cross_correlation_2d(img, sobel_y)

    mag = np.sqrt(grad_x**2 + grad_y**2)
    dir = np.arctan2(grad_y, grad_x)

    return mag, dir





if __name__=="__main__":
    import time

    lenna = cv2.imread('./A1_Images/lenna.png', cv2.IMREAD_GRAYSCALE)
    shapes = cv2.imread('./A1_Images/shapes.png', cv2.IMREAD_GRAYSCALE)
    lenna = np.asarray(lenna).astype(np.float32)
    shapes = np.asarray(shapes).astype(np.float32)

    filter = get_gaussian_filter_2d(7, 1.5)
    filtered_lenna = cross_correlation_2d(lenna, filter)
    filtered_shapes = cross_correlation_2d(shapes, filter)

    start = time.time()
    lenna_mag, lenna_dir = compute_image_gradient(filtered_lenna)
    print(f"Lenna - Computational Time of Applying Sobel filter: {time.time() - start:.5f} sec")

    cv2.imwrite('./result/part_2_edge_raw_lenna.png', np.clip(lenna_mag, 0, 255).astype(np.uint8))
    cv2.imshow("Sobel Filtered lenna.png", np.clip(lenna_mag, 0, 255).astype(np.uint8))
    cv2.waitKey(1000)

    start = time.time()
    shapes_mag, shapes_dir = compute_image_gradient(filtered_shapes)
    print(f"Shapes - Computational Time of Applying Sobel filter: {time.time() - start:.5f} sec")

    cv2.imwrite('./result/part_2_edge_raw_shapes.png', np.clip(shapes_mag, 0, 255).astype(np.uint8))
    cv2.imshow("Sobel Filtered shapes.png", np.clip(shapes_mag, 0, 255).astype(np.uint8))
    cv2.waitKey(1000)