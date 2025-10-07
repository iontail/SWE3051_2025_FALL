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
    n = 5*5

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
    response = (response - r_min) / ((r_max - r_min) + 1e-6) # avoid division by zero
    return response


def non_maximum_suppresion_win(R: np.ndarray, winSize: int = 11):
    h, w = R.shape
    h_pad_size = (winSize - 1) // 2
    w_pad_size = (winSize - 1) // 2

    padded_R = pad_img(R, h_pad_size, w_pad_size, zero_pad=True)

    out = np.zeros_like(R, dtype=R.dtype)
    for i in range(h):
        for j in range(w):
            center = padded_R[i + h_pad_size, j + w_pad_size]
            if not(center > 0.1):
                continue

            patch = padded_R[i:i + winSize, j:j + winSize]

            maximum = patch.max()
            if maximum > center:
                continue
            out[i, j] = 1
    return out 


if __name__=="__main__":
    import time

    lenna = cv2.imread('./A1_Images/lenna.png', cv2.IMREAD_GRAYSCALE)
    shapes = cv2.imread('./A1_Images/shapes.png', cv2.IMREAD_GRAYSCALE)
    lenna = np.asarray(lenna).astype(np.float32)
    shapes = np.asarray(shapes).astype(np.float32)

    # 3-1
    filter = get_gaussian_filter_2d(7, 1.5)
    filtered_lenna = cross_correlation_2d(lenna, filter)
    filtered_shapes = cross_correlation_2d(shapes, filter)

    # ====== Lenna ======
    # 3-2-e
    start = time.time()
    lenna_response = compute_corner_response(filtered_lenna)
    print(f"Lenna - Computational Time of Computing Corner Response: {time.time() - start:.5f} sec")
    cv2.imwrite('./result/part_3_corner_raw_lenna.png', np.clip(lenna_response * 255, 0, 255).astype(np.uint8))
    cv2.imshow("Response of lenna.png", np.clip(lenna_response * 255, 0, 255).astype(np.uint8))
    cv2.waitKey(1)

    # 3-3-b
    lenna_bgr = cv2.cvtColor(lenna.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    threshold_mask = lenna_response > 0.1
    lenna_corner = lenna_bgr.copy()
    lenna_corner[threshold_mask] = (0, 255, 0)

    cv2.imwrite('./result/part_3_corner_bin_lenna.png', lenna_corner)
    cv2.imshow("Response Dot of lenna.png", lenna_corner)
    cv2.waitKey(1)

    # 3-3-d
    start = time.time()
    lenna_corner_detection = non_maximum_suppresion_win(lenna_response)
    print(f"Lenna - Computational Time of NMS Window: {time.time() - start:.5f} sec")
    ys, xs = np.where(lenna_corner_detection != 0)
    for y, x in zip(ys, xs):
        cv2.circle(lenna_bgr, (x, y), 5, (0, 255, 0), 2)

    cv2.imwrite('./result/part_3_corner_sup_lenna.png', lenna_bgr)
    cv2.imshow("Response Dot of lenna.png", lenna_bgr)
    cv2.waitKey(1)

    
    # ====== Shapes ======
    # 3-2-e
    start = time.time()
    shapes_response = compute_corner_response(filtered_shapes)
    print(f"Shapes - Computational Time of Computing Corner Response: {time.time() - start:.5f} sec")
    cv2.imwrite('./result/part_3_corner_raw_shapes.png', np.clip(shapes_response * 255, 0, 255).astype(np.uint8))
    cv2.imshow("Response of shapes.png", np.clip(shapes_response * 255, 0, 255).astype(np.uint8))
    cv2.waitKey(1)

    # 3-3-b
    shapes_bgr = cv2.cvtColor(shapes.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    threshold_mask = shapes_response > 0.1
    shapes_corner = shapes_bgr.copy()
    shapes_corner[threshold_mask] = (0, 255, 0)

    cv2.imwrite('./result/part_3_corner_bin_shapes.png', shapes_corner)
    cv2.imshow("Response Dot of shapes.png", shapes_corner)
    cv2.waitKey(1)

    # 3-3-d
    start = time.time()
    shapes_corner_detection = non_maximum_suppresion_win(shapes_response)
    print(f"Shapes - Computational Time of NMS Window: {time.time() - start:.5f} sec")
    ys, xs = np.where(shapes_corner_detection != 0)
    for y, x in zip(ys, xs):
        cv2.circle(shapes_bgr, (x, y), 5, (0, 255, 0), 2)

    cv2.imwrite('./result/part_3_corner_sup_shapes.png', shapes_bgr)
    cv2.imshow("Response Dot of shapes.png", shapes_bgr)
    cv2.waitKey(0)

