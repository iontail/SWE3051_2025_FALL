import cv2
import numpy as np
import math

from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d, pad_img
from A1_edge_detection import compute_image_gradient, non_maximum_suppression_dir


def compute_second_moment(grad_x: np.ndarray, grad_y: np.ndarray):

    Ixx = grad_x * grad_x
    Ixy = grad_x * grad_y
    Iyy = grad_y * grad_y

    window = np.ones((5, 5), dtype=grad_x.dtype)

    xx = cross_correlation_2d(Ixx, kernel=window, zero_pad=True)
    xy = cross_correlation_2d(Ixy, kernel=window, zero_pad=True)
    yy = cross_correlation_2d(Iyy, kernel=window, zero_pad=True)

    return xx, xy, yy

def compute_corner_response(img: np.ndarray):

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

    img = np.clip(img, 0, 255)
    grad_x = cross_correlation_2d(img, sobel_x)
    grad_y = cross_correlation_2d(img, sobel_y)

    xx, xy, yy = compute_second_moment(grad_x, grad_y)

    det = (xx * yy) - (xy**2)
    tr = xx + yy

    response = det - 0.04  * (tr**2)
    response[response < 0] = 0

    r_min, r_max = response.min(), response.max()
    response = (response - r_min / (r_max - r_min))


    return response





if __name__=="__main__":
    import time

    lenna = cv2.imread('./A1_Images/lenna.png', cv2.IMREAD_GRAYSCALE)
    shapes = cv2.imread('./A1_Images/shapes.png', cv2.IMREAD_GRAYSCALE)
    lenna = np.asarray(lenna).astype(np.float32)
    shapes = np.asarray(shapes).astype(np.float32)

    filter = get_gaussian_filter_2d(7, 1.5)
    filtered_lenna = cross_correlation_2d(lenna, filter)
    filtered_shapes = cross_correlation_2d(shapes, filter)

    
    # ====== Lenna ======
    start = time.time()
    lenna_response = compute_corner_response(lenna)
    print(lenna_response)
    print(f"Lenna - Computational Time of Computing Corner Response: {time.time() - start:.5f} sec")

