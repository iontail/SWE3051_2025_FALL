import cv2
import numpy as np
import sys

def get_transformed_image(img: np.ndarray, m: np.ndarray):
    h, w = img.shape
    half_h = h//2
    half_w = w//2

    t_plane = np.full((801, 801), 255, dtype=img.dtype)
    for i in range(h):
        for j in range(w):
            x = j - half_w
            y = half_h - i

            t_x, t_y, t_z = np.dot(m, np.array([x, y, 1], dtype=np.float32))
            t_x = int(t_x / t_z) + 400
            t_y = 400 - int(t_y / t_z)

            if 0 <= t_x < 801 and 0 <= t_y < 801:
                t_plane[t_y, t_x] = img[i, j]

    return t_plane


def get_transformation_matrix(key: str):

    # as numpy array has filped y-axis
    # I multiply -1 to the elements related to y-axis

    if key == 'a':
        m = np.array([[1, 0, -5], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'd':
        m = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'w':
        m = np.array([[1, 0, 0], [0, 1, 5], [0, 0, 1]], dtype=np.float32)

    elif key == 's':
        m = np.array([[1, 0, 0], [0, 1, -5], [0, 0, 1]], dtype=np.float32)

    elif key == 'r':
        cos = np.cos(np.deg2rad(5))
        sin = np.sin(np.deg2rad(5))

        m = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 't':
        cos = np.cos(np.deg2rad(-5))
        sin = np.sin(np.deg2rad(-5))

        m = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'f':
        m = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'g':
        m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'x':
        # TODO: Is sequential shrinking operation is multiplicative or additive?
        m = np.array([[0.95, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'c':
        m = np.array([[1.05, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'y':
        m = np.array([[1, 0, 0], [0, 0.95, 0], [0, 0, 1]], dtype=np.float32)
        
    elif key == 'u':
        m = np.array([[1, 0, 0], [0, 1.05, 0], [0, 0, 1]], dtype=np.float32)
    else:
        m = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    return m


def visualize_transformation(transformed_img: np.ndarray, name: str = 'smile'):
    cv2.arrowedLine(transformed_img, (0,400), (800,400), (0, 0, 0), 2, tipLength=0.015)
    cv2.arrowedLine(transformed_img, (400,800), (400,0), (0, 0, 0), 2, tipLength=0.015)
    cv2.imshow(f'Transformed Image - {name}', transformed_img.astype(np.uint8))

if __name__ == "__main__":
    smile = cv2.imread('./A2_Images/smile.png', cv2.IMREAD_GRAYSCALE)
    smile = np.asarray(smile, dtype=np.float32)

    m_total = get_transformation_matrix('initial')
    transformed_img = get_transformed_image(smile, m_total)
    while True:
        visualize_transformation(transformed_img, 'smile')
        key = cv2.waitKey()

        if key == ord('q'):
            cv2.destroyAllWindows()
            break

        elif key == ord('h'):
            m_total = get_transformation_matrix('initial')
            transformed_img = get_transformed_image(smile, m_total)
        else:
            m = get_transformation_matrix(chr(key))
            m_total = np.dot(m, m_total)
        transformed_img = get_transformed_image(smile, m_total)

