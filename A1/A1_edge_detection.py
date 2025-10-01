import cv2
import numpy as np
import math

from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d, pad_img

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

    img = np.clip(img, 0, 255)
    grad_x = cross_correlation_2d(img, sobel_x)
    grad_y = cross_correlation_2d(img, sobel_y)

    mag = np.sqrt(grad_x**2 + grad_y**2)
    dir = np.arctan2(grad_y, grad_x)

    return mag, dir

def quantize_gradient_dir(dir: np.ndarray):
    # dir range: [-pi, pi]
    deg = np.degrees(dir)
    deg = (deg + 360) % 360
    bins = (((deg + 22.5) // 45.0) % 8).astype(np.int32)
    return bins

def non_maximum_suppression_dir(mag: np.ndarray, dir: np.ndarray):
    h, w = mag.shape
    pos_dir = {
        0: (0, 1),
        1: (1, 1),
        2: (1, 0),
        3: (1, -1),
        4: (0, -1),
        5: (-1, -1),
        6: (-1, 0),
        7: (-1, 1)
    }

    q_dir = quantize_gradient_dir(dir)

    # TODO: how to handle if the comparison pixel does not exit?
    # I simply padded image
    padded_mag = pad_img(mag, 1, 1)

    out = mag.copy() 
    for i in range(h):
        for j in range(w):
            dy, dx = pos_dir[q_dir[i, j]]

            center = padded_mag[i + 1, j + 1]
            comparison_1 = padded_mag[i + 1 + dy, j + 1 + dx]
            comparison_2 = padded_mag[i + 1 - dy, j + 1 - dx]
            

            if center < comparison_1 or center < comparison_2:
                out[i, j] = 0

    return out
                

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
    lenna_mag, lenna_dir = compute_image_gradient(filtered_lenna)
    print(f"Lenna - Computational Time of Applying Sobel filter: {time.time() - start:.5f} sec")

    cv2.imwrite('./result/part_2_edge_raw_lenna.png', np.clip(lenna_mag, 0, 255).astype(np.uint8))
    cv2.imshow("Sobel Filtered lenna.png", np.clip(lenna_mag, 0, 255).astype(np.uint8))
    cv2.waitKey(1000)

    start = time.time()
    lenna_nms = non_maximum_suppression_dir(lenna_mag, lenna_dir)
    print(f"Lenna - Computational Time of NMS: {time.time() - start:.5f} sec")

    cv2.imwrite('./result/part_2_edge_sup_lenna.png', np.clip(lenna_nms, 0, 255).astype(np.uint8))
    cv2.imshow("NMS lenna.png", np.clip(lenna_mag, 0, 255).astype(np.uint8))
    cv2.waitKey(1000)


    # ====== Shapes ======
    start = time.time()
    shapes_mag, shapes_dir = compute_image_gradient(filtered_shapes)
    print(f"Shapes - Computational Time of Applying Sobel filter: {time.time() - start:.5f} sec")

    cv2.imwrite('./result/part_2_edge_raw_shapes.png', np.clip(shapes_mag, 0, 255).astype(np.uint8))
    cv2.imshow("Sobel Filtered shapes.png", np.clip(shapes_mag, 0, 255).astype(np.uint8))
    cv2.waitKey(1000)

    start = time.time()
    shapes_nms = non_maximum_suppression_dir(shapes_mag, shapes_dir)
    print(f"Shapes - Computational Time of NMS: {time.time() - start:.5f} sec")

    cv2.imwrite('./result/part_2_edge_sup_shapes.png', np.clip(shapes_nms, 0, 255).astype(np.uint8))
    cv2.imshow("NMS shapes.png", np.clip(shapes_nms, 0, 255).astype(np.uint8))
    cv2.waitKey(1000)

