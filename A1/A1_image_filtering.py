import cv2
import numpy as np

def pad_img(img: np.array, h_pad_size:int, w_pad_size: int):
    h, w = img.shape
    padded_image = np.zeros([h + h_pad_size * 2, w + w_pad_size * 2], dtype=img.dtype)

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




def cross_correlation_1d(img, kernel):
    img = np.asarray(img)
    kernel = np.asarray(img)

    h, w = img.shape
    kenel_size = kernel.shape

    if kernel.shape[0]==1:
        horizontal = True
    elif kernel.shape[1]==1:
        vertical = True
    else:
        raise ValueError(f'cross_correlation_1d function kernel should be vertical or horizontal. Got {kernel.shape} shape kernel.')

    # from the formula: '(img_size - kernel_size + 2 * padding)/stride + 1 = output_size',
    # the formula becomes 'padding = (kerne_size - 1) // 2' (cosider only odd size case)
    h_pad_size = (kernel[0] - 1) // 2  
    w_pad_sie =  (kenel_size[0] - 1) // 2

    if horizontal:
        padded_img = None

    elif vertical:
        padded_img = None


def cross_correlation_1d(img, kernel):
    """
    Assumes kernel size is odd.
    Args:
        img (tuple): (height, width) tuple
        kernel (tuble): (kernel_height, kernel_width) tupel

    Returns:
        filtered_img (tuple): filtered image with given kernel 
    """


def get_gaussian_filter_1d(size, sigma):
    pass

def get_gaussian_filter_2d(size, sigma):
    pass


if __name__=="__main__":
    img = cv2.imread('./A1_Images/lenna.png', cv2.IMREAD_GRAYSCALE)
    print(img.shape[0])
    np_img = np.asarray(img)
    padded_img = pad_img(np_img, 50, 50)
    print(padded_img.shape)

    cv2.imshow('Transformed Image', padded_img)
    cv2.waitKey(0)

    