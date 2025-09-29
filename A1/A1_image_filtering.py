import cv2

def cross_correlation_1d(img, kernel):
    h, w = img.shape
    kenel_size = kernel.shape

    if kernel.shape[0]==1:
        horizontal = True
    elif kernel.shape[1]==1:
        vertical = True
    else:
        raise ValueError(f'cross_correlation_1d function kernel should be vertical or horizontal. Got {kernel.shape} shape kernel.')

    # from the formula: (img_size - kernel_size + 2 * padding)/stride + 1 = output_size
    # The formula becoes (kerne_size - 1) // 2 == padding (cosider only odd size case)
    h_pad_size = (kernel[0] - 1) // 2  
    w_pad_sie =  (kenel_size[0] - 1) // 2


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

    cv2.imshow('Transformed Image', img)
    cv2.waitKey(0)

    