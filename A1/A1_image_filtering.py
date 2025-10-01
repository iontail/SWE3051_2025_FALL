import cv2
import numpy as np
import time

def pad_img(img: np.ndarray, h_pad_size:int, w_pad_size: int):
    h, w = img.shape
    padded_image = np.zeros((h + h_pad_size * 2, w + w_pad_size * 2), dtype=img.dtype)

    padded_image[h_pad_size:h_pad_size + h, w_pad_size:w_pad_size + w] = img[:, :]

    if h_pad_size > 0:
        padded_image[:h_pad_size, w_pad_size:w_pad_size + w] = img[0:1, :]
        padded_image[h_pad_size + h:, w_pad_size:w_pad_size + w] = img[-1:, :]

    if w_pad_size > 0:
        padded_image[h_pad_size:h_pad_size + h, :w_pad_size] = img[:, 0:1]
        padded_image[h_pad_size:h_pad_size + h, w_pad_size + w:] = img[:, -1:]

    if h_pad_size > 0 and w_pad_size > 0: # pad corner
        padded_image[:h_pad_size, :w_pad_size] = img[0, 0]
        padded_image[:h_pad_size, w_pad_size + w:] = img[0, -1]
        padded_image[h_pad_size + h:, :w_pad_size] = img[-1, 0]
        padded_image[h_pad_size + h:, w_pad_size + w:] = img[-1, -1]
        

    return padded_image

def cross_correlation_1d(img: np.ndarray, kernel: np.ndarray):
    h, w = img.shape
    kernel_size = kernel.shape
    

    # From the formula: '(img_size - kernel_size + 2 * padding)/stride + 1 = output_size', 
    # the formula becomes 'padding = (kerne_size - 1) // 2' 
    # when output_size == img_size (cosider only odd size case)
    h_pad_size = (kernel_size[0] - 1) // 2  
    w_pad_size =  (kernel_size[1] - 1) // 2

    padded_img = pad_img(img, h_pad_size, w_pad_size)
    
    filtered_img = np.zeros((h, w), dtype=img.dtype)

    kernel = kernel.reshape(-1) # flatten
    if kernel_size[0] == 1: # horizontal kernel
        for i in range(h):
            for j in range(w):
                patch = padded_img[i, j:j + kernel_size[1]].reshape(-1)
                filtered_img[i, j] = np.sum(patch * kernel)

    else: # vertical kernel
        for i in range(h):
            for j in range(w):
                patch = padded_img[i:i + kernel_size[0], j].reshape(-1)
                filtered_img[i, j] = np.sum(patch * kernel)

    return filtered_img


def cross_correlation_2d(img: np.ndarray, kernel: np.ndarray):
    h, w = img.shape
    kernel_size = kernel.shape
    
    h_pad_size = (kernel_size[0] - 1) // 2  
    w_pad_size =  (kernel_size[1] - 1) // 2

    padded_img = pad_img(img, h_pad_size, w_pad_size)
    
    filtered_img = np.zeros((h, w), dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            patch = padded_img[i:i + kernel_size[0], j:j + kernel_size[1]]
            filtered_img[i, j] = np.sum(patch * kernel)

    return filtered_img

def get_gaussian_filter_1d(size: int, sigma: float):

    r = size // 2
    idx = np.arange(-r, r+1, dtype=np.float32)

    # constant term is mutiplied to every filter entries
    # as I normalized it, constant term can be dismissed
    gaussian = np.exp(-(idx**2) / (2 * sigma**2))
    filter = gaussian / gaussian.sum()

    return filter.reshape(1, -1)


def get_gaussian_filter_2d(size: int, sigma: float):
    # I only need 1d gaussian filters as 2d gaussian filter is seperable

    horizontal_filter = get_gaussian_filter_1d(size, sigma)
    gaussian = horizontal_filter.T @ horizontal_filter
    return gaussian 

def visualize_filtering(
        img: np.ndarray,
        kerenl_size: list[int],
        sigma_list: list[int],
        name: str
        ):
    
    row = []
    for k in kerenl_size:
        row_imgs = []
        for s in sigma_list:
            filter = get_gaussian_filter_2d(k, s)
            filtered_img = np.clip(cross_correlation_2d(img, filter), 0, 255).astype(np.uint8)

            text = f"{k}x{k} s={s}"
            cv2.putText(filtered_img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            row_imgs.append(filtered_img)

        row.append(np.hstack(row_imgs))
    out = np.vstack(row)

    cv2.imwrite(f'./result/part_1_gaussian_filtered_{name}.png', out)
    return out

def visualize_filtering_difference(
        img: np.ndarray,
        kerenl_size: list[int],
        sigma_list: list[int]
        ):
    
    row = []
    difs = []
    times_1d = []
    times_2d = []
    for k in kerenl_size:
        row_imgs = []
        for s in sigma_list:
            filter_2d = get_gaussian_filter_2d(k, s)
            start = time.time()
            filtered_img_2d_filter = np.clip(cross_correlation_2d(img, filter_2d), 0, 255)
            times_2d.append(time.time() - start)
            

            filter_1d = get_gaussian_filter_1d(k, s)
            start = time.time()
            filtered_img_1d_filter = cross_correlation_1d(img, filter_1d.T)
            filtered_img_1d_filter = np.clip(cross_correlation_1d(filtered_img_1d_filter, filter_1d), 0, 255)
            times_1d.append(time.time() - start)

            dif = np.abs(filtered_img_2d_filter - filtered_img_1d_filter)
            dif_sum = dif.sum()
            difs.append(dif_sum)

            text = f"{k}x{k} s={s}"
            dif = dif.astype(np.uint8)
            cv2.putText(dif, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            row_imgs.append(dif)
            
        row.append(np.hstack(row_imgs))
    out = np.vstack(row)

    difs = np.array(difs).reshape(len(kerenl_size), len(sigma_list))
    times_1d = np.array(times_1d).reshape(len(kerenl_size), len(sigma_list))
    times_2d = np.array(times_2d).reshape(len(kerenl_size), len(sigma_list))
    return out, difs, times_1d, times_2d


if __name__=="__main__":
    import time
    lenna = cv2.imread('./A1_Images/lenna.png', cv2.IMREAD_GRAYSCALE)
    shapes = cv2.imread('./A1_Images/shapes.png', cv2.IMREAD_GRAYSCALE)
    lenna = np.asarray(lenna).astype(np.float32)
    shapes = np.asarray(shapes).astype(np.float32)


    print(f"Gaussian Filter 1d (5, 1):\n{get_gaussian_filter_1d(5, 1)}")
    print(f"Gaussian Filter 2d (5, 1):\n{get_gaussian_filter_2d(5, 1)}")


    lenna_dif_output, lenna_difs, times_1d, times_2d = visualize_filtering_difference(lenna, [5, 11, 17], [1, 6, 11])
    print(f"Lenna - Computational time of Seperable filter:\n{times_1d}")
    print(f"Lenna - Computational time of non-Seperable filter:\n{times_2d}")
    print(f"Lenna - Abosolute different summation:\n{lenna_difs}")
    cv2.imshow("Lenna - Gaussian Filtered Images Difference", lenna_dif_output)
    cv2.waitKey(1000)

    shapes_dif_output, shapes_difs, times_1d, times_2d = visualize_filtering_difference(shapes, [5, 11, 17], [1, 6, 11])
    print(f"Shapes - Computational time of Seperable filter:\n{times_1d}")
    print(f"shapes - Computational time of non-Seperable filter:\n{times_2d}")
    print(f"Shapes - Abosolute different summation:\n{shapes_difs}")
    cv2.imshow("Shapes - Gaussian Filtered Images Difference", shapes_dif_output)
    cv2.waitKey(0)