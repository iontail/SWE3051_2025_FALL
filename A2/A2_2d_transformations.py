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
            y = half_h - i # flip y-axis to match image y-coordinate

            t_x, t_y, t_z = np.dot(m, np.array([x, y, 1], dtype=np.float32))
            t_x = int(t_x / t_z) + 400
            t_y = 400 - int(t_y / t_z)

            if 0 <= t_x < 801 and 0 <= t_y < 801:
                t_plane[t_y, t_x] = img[i, j]

    return t_plane


def get_transformation_matrix(key: str, repeats: int = 1):

    if key == 'a':
        pixel = -5 * repeats
        m = np.array([[1, 0, pixel], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'd':
        pixel = 5 * repeats
        m = np.array([[1, 0, pixel], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'w':
        pixel = 5 * repeats
        m = np.array([[1, 0, 0], [0, 1, pixel], [0, 0, 1]], dtype=np.float32)

    elif key == 's':
        pixel = -5 * repeats
        m = np.array([[1, 0, 0], [0, 1, pixel], [0, 0, 1]], dtype=np.float32)

    elif key == 'r':
        cos = np.cos(np.deg2rad(5 * repeats))
        sin = np.sin(np.deg2rad(5 * repeats))

        m = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 't':
        cos = np.cos(np.deg2rad(-5 * repeats))
        sin = np.sin(np.deg2rad(-5 * repeats))

        m = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'f':
        m = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'g':
        m = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'x':
        # TODO: Is sequential shrinking operation is multiplicative or additive?
        shrink_factor = 1.0 - (0.05 * repeats)
        m = np.array([[shrink_factor, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'c':
        enlarge_factor = 1.0 + (0.05 * repeats)
        m = np.array([[enlarge_factor, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    elif key == 'y':
        shrink_factor = 1.0 - (0.05 * repeats)
        m = np.array([[1, 0, 0], [0, shrink_factor, 0], [0, 0, 1]], dtype=np.float32)
        
    elif key == 'u':
        enlarge_factor = 1.0 + (0.05 * repeats)
        m = np.array([[1, 0, 0], [0, enlarge_factor, 0], [0, 0, 1]], dtype=np.float32)

    return m

def apply_transformations(img: np.ndarray, prompt: str):
    initial_state = img.copy()
    for p in prompt:
        current_p = p.strip()

        if 48 <= ord(current_p[0]) <= 57:
            repeats = int(current_p[:-1])
            key = current_p[-1]
        else:
            repeats = 1
            key = current_p[-1]

        # non-transform keys
        if key == 'h':
            img = initial_state.copy()
        elif key == 'q':
            print("\'Quit\' command received. Terminate the transformation process.")
            cv2.destroyAllWindows()
            break
        else:
            m = get_transformation_matrix(key=key, repeats=repeats)
            img = get_transformed_image(img, m)

    return img

def visualize_transformation(transformed_img: np.ndarray, name: str = 'smile'):
    cv2.arrowedLine(transformed_img, (0,400), (801,400), (0, 0, 0), 2, tipLength=0.015)
    cv2.arrowedLine(transformed_img, (400,801), (400,0), (0, 0, 0), 2, tipLength=0.015)
    cv2.imshow(f'Transformed Image - {name}', transformed_img.astype(np.uint8))
    cv2.waitKey(0)


if __name__ == "__main__":
    smile = cv2.imread('./A2_Images/smile.png', cv2.IMREAD_GRAYSCALE)
    smile = np.asarray(smile, dtype=np.float32)

    prompt = sys.stdin.readline().strip().split('+')
    transformed_img = apply_transformations(smile, prompt)
    visualize_transformation(transformed_img, 'smile')