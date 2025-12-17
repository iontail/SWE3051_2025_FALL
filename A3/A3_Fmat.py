import time
import cv2
import numpy as np


# storing image sizes globally
_W1 = _H1 = _W2 = _H2 = None
def set_image_sizes(w1: int, h1: int, w2: int, h2: int):
    global _W1, _H1, _W2, _H2
    _W1, _H1, _W2, _H2 = w1, h1, w2, h2


def compute_F_raw(M: np.ndarray):
    x1, y1, x2, y2 = M[:, 0], M[:, 1], M[:, 2], M[:, 3]

    one = np.ones_like(x1)
    A = np.stack([
        x2 * x1, x2 * y1, x2,
        y2 * x1, y2 * y1, y2,
        x1, y1, one
        ], axis=1)
    

    # compute F using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1, :]
    F = F.reshape(3, 3)

    # Rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    F = U @ np.diag(S) @ Vt
    return F


def _build_T(h: int, w: int):
    center_x = (w - 1) / 2.0
    center_y = (h - 1) / 2.0

    scale_x = 2.0 / (w - 1)
    scale_y = 2.0 / (h - 1)
    
    T = np.array([
        [scale_x, 0, -scale_x * center_x],
        [0, scale_y, -scale_y * center_y],
        [0, 0, 1]], 
        dtype=np.float32
    )
    return T

def compute_F_norm(M: np.ndarray):
    T1 = _build_T(_H1, _W1)
    T2 = _build_T(_H2, _W2)

    n = M.shape[0] 
    x1 = np.stack([M[:,0], M[:,1], np.ones(n)], axis=1) # (n, 3) = (n, (x, y, 1))
    x2 = np.stack([M[:,2], M[:,3], np.ones(n)], axis=1)

    x1n = (T1 @ x1.T).T # (n, 3)
    x2n = (T2 @ x2.T).T
    M_norm = np.stack([x1n[:,0], x1n[:,1], x2n[:,0], x2n[:,1]], axis=1) # (n, 4)

    F_n = compute_F_raw(M_norm)          
    F = T2.T @ F_n @ T1     
    return F


# ======= compute_F_mine ===========
def compute_distance(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """
    x1, x2 should be in homogeneous coordinates
    """
    # epipolar line in image 2
    l2 = (F @ x1.T).T  # (N,3)

    num = np.abs(np.sum(l2 * x2, axis=1))
    denom = np.sqrt(l2[:,0]**2 + l2[:,1]**2) + 1e-9
    return num / denom

def compute_F_mine(M: np.ndarray, th: float = 1.0):
    start_time = time.time()

    n = M.shape[0]

    # homogeneous coordinates
    x1 = np.stack([M[:, 0], M[:, 1], np.ones(n)], axis=1)  # (n,3)
    x2 = np.stack([M[:, 2], M[:, 3], np.ones(n)], axis=1)  # (n,3)

    best_inlier = None
    best_cnt = 0

    while time.time() - start_time < 4.9:
        idx = np.random.choice(n, 8, replace=False)
        M_sample = M[idx]

        F = compute_F_norm(M_sample)

        # point-line distance
        error = compute_distance(F, x1, x2)

        inlier = np.where(error < th)[0]

        if len(inlier) > best_cnt:
            best_cnt = len(inlier)
            best_inlier = inlier

    # final F computation
    M_inlier = M[best_inlier]
    F_final = compute_F_norm(M_inlier)
    return F_final

def show_error(M: np.ndarray, name: str, th: float = 1.0, format: str = 'png'):
    n = M.shape[0]
    
    x1 = np.stack([M[:, 0], M[:, 1], np.ones(n)], axis=1)  # (n,3)
    x2 = np.stack([M[:, 2], M[:, 3], np.ones(n)], axis=1)  # (n,3)

    raw_F = compute_F_raw(M)
    norm_F = compute_F_norm(M)
    mine_F = compute_F_mine(M, th)
    error_raw = compute_distance(raw_F, x1, x2)
    error_norm = compute_distance(norm_F, x1, x2)
    error_mine = compute_distance(mine_F, x1, x2)

    print(f"Average Reprojection Errors ({name}1.{format} and {name}2.{format})")
    print(f"  {'Raw':<7} = {np.mean(error_raw)}")
    print(f"  {'Norm':<7} = {np.mean(error_norm)}")
    print(f"  {'Mine':<7} = {np.mean(error_mine)}")


if __name__ == '__main__':
    
    temple1 = cv2.imread('./A3_P1_Data/temple1.png', cv2.IMREAD_GRAYSCALE)
    temple1 = np.asarray(temple1, dtype=np.float32)
    temple2 = cv2.imread('./A3_P1_Data/temple2.png', cv2.IMREAD_GRAYSCALE)
    temple2 = np.asarray(temple2, dtype=np.float32)

    M = np.loadtxt('./A3_P1_Data/temple_matches.txt')
    set_image_sizes(temple1.shape[1], temple1.shape[0], temple2.shape[1], temple2.shape[0])
    show_error(M, 'temple', th=1.0)


    house1 = cv2.imread('./A3_P1_Data/house1.jpg', cv2.IMREAD_GRAYSCALE)
    house1 = np.asarray(house1, dtype=np.float32)
    house2 = cv2.imread('./A3_P1_Data/house2.jpg', cv2.IMREAD_GRAYSCALE)
    house2 = np.asarray(house2, dtype=np.float32)

    M = np.loadtxt('./A3_P1_Data/house_matches.txt')
    set_image_sizes(house1.shape[1], house1.shape[0], house2.shape[1], house2.shape[0])
    show_error(M, 'house', th=1.0, format='jpg')

    library1 = cv2.imread('./A3_P1_Data/library1.jpg', cv2.IMREAD_GRAYSCALE)
    library1 = np.asarray(library1, dtype=np.float32)
    library2 = cv2.imread('./A3_P1_Data/library2.jpg', cv2.IMREAD_GRAYSCALE)
    library2 = np.asarray(library2, dtype=np.float32)

    M = np.loadtxt('./A3_P1_Data/library_matches.txt')
    set_image_sizes(library1.shape[1], library1.shape[0], library2.shape[1], library2.shape[0])
    show_error(M, 'library', th=1.0, format='jpg')


    
