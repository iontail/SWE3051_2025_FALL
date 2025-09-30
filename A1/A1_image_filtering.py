import cv2
import numpy as np

def pad_img(img: np.array, h_pad_size:int, w_pad_size: int):
    h, w = img.shape
    padded_image = np.zeros((h + h_pad_size * 2, w + w_pad_size * 2), dtype=img.dtype)

    padded_image[h_pad_size:h_pad_size + h, w_pad_size:w_pad_size + w] = img[:, :]

    if h_pad_size > 0:
        padded_image[:h_pad_size, w_pad_size:w_pad_size + w] = img[0:1, :]
        padded_image[h_pad_size + h:, w_pad_size:w_pad_size + w] = img[-1:, :]

    if w_pad_size > 0:
        padded_image[h_pad_size:h_pad_size + h, :w_pad_size] = img[:, 0:1]
        padded_image[h_pad_size:h_pad_size + h, w_pad_size + w:] = img[:, -1:]

    if h_pad_size > 0 and w_pad_size: # pad corner
        padded_image[:h_pad_size, :w_pad_size] = img[0, 0]
        padded_image[:h_pad_size, w_pad_size + w:] = img[0, -1]
        padded_image[h_pad_size + h:, :w_pad_size] = img[-1, 0]
        padded_image[h_pad_size + h:, w_pad_size + w:] = img[-1, -1]
        

    return padded_image

def cross_correlation_1d(img: np.array, kernel: np.array):
    h, w = img.shape
    kernel_size = kernel.shape
    

    # from the formula: '(img_size - kernel_size + 2 * padding)/stride + 1 = output_size',
    # the formula becomes 'padding = (kerne_size - 1) // 2' (cosider only odd size case)
    h_pad_size = (kernel_size[0] - 1) // 2  
    w_pad_size =  (kernel_size[1] - 1) // 2

    padded_img = pad_img(img, h_pad_size, w_pad_size)
    
    filtered_img = np.zeros((h, w), dtype=img.dtype)
    if kernel_size[0] == 1: # horizontal kernel
        for i in range(h):
            for j in range(w):
                patch = padded_img[i + h_pad_size, j:j + kernel_size[1]]
                filtered_img[i, j] = np.sum(patch * kernel)

    else: # vertical kernel
        for i in range(h):
            for j in range(w):
                patch = padded_img[i:i + kernel_size[0], j + w_pad_size]
                filtered_img[i, j] = np.sum(patch * kernel)

    return np.clip(filtered_img, 0, 255)


def cross_correlation_2d(img: np.array, kernel: np.array):
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

    return np.clip(filtered_img, 0, 255)

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



if __name__=="__main__":
    img = cv2.imread('./A1_Images/lenna.png', cv2.IMREAD_GRAYSCALE)
    print(img.shape[0])
    np_img = np.asarray(img)
    np_img = np_img.astype(np.float32)

    
    #kernel = np.array([1/25 for _ in range(25)]).reshape(5, 5)
    kernel = get_gaussian_filter_1d(10, 10)
    print(kernel.sum())
    filtered_img = cross_correlation_1d(np_img, kernel)

    cv2.imshow('Filtered Image', filtered_img.astype(np.uint8))
    cv2.waitKey(0)





    