import time
import cv2
import random
import numpy as np
from A3_P1_Data.compute_avg_reproj_error import compute_avg_reproj_error



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
    error_raw = compute_avg_reproj_error(M, raw_F)
    error_norm = compute_avg_reproj_error(M, norm_F)
    error_mine = compute_avg_reproj_error(M, mine_F)

    print(f"Average Reprojection Errors ({name}1.{format} and {name}2.{format})")
    print(f"  {'Raw':<7} = {np.mean(error_raw)}")
    print(f"  {'Norm':<7} = {np.mean(error_norm)}")
    print(f"  {'Mine':<7} = {np.mean(error_mine)}")
    return raw_F, norm_F, mine_F

# ====== Visualization ======
def line_endpoints_in_image(line_abc: np.ndarray, w: int, h: int):
    """
    line_abc: (3,) where a x + b y + c = 0
    Return:
        ((x0,y0),(x1,y1)) as int tuples
    """
    a, b, c = float(line_abc[0]), float(line_abc[1]), float(line_abc[2])

    pts = []

    # x=0, x=w-1
    if abs(b) > 0:
        y0 = -(c + a * 0.0) / b
        y1 = -(c + a * (w - 1.0)) / b
        if 0.0 <= y0 <= (h - 1.0):
            pts.append((0.0, y0))
        if 0.0 <= y1 <= (h - 1.0):
            pts.append((w - 1.0, y1))

    # y=0, y=h-1
    if abs(a) > 0:
        x0 = -(c + b * 0.0) / a
        x1 = -(c + b * (h - 1.0)) / a
        if 0.0 <= x0 <= (w - 1.0):
            pts.append((x0, 0.0))
        if 0.0 <= x1 <= (w - 1.0):
            pts.append((x1, h - 1.0))

    # pick two farthest points
    best = None
    best_d = -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = (pts[i][0] - pts[j][0])**2 + (pts[i][1] - pts[j][1])**2
            if d > best_d:
                best_d = d
                best = (pts[i], pts[j])

    (x0, y0), (x1, y1) = best
    return (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1)))


def draw_epipolar_demo(img1: np.ndarray, img2: np.ndarray, M: np.ndarray, F: np.ndarray, win_name="Epipolar"):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    colors = [(0,0,255), (0,255,0), (255,0,0)]  # BGR: red, green, blue

    n = M.shape[0]
    while True:
        vis1 = img1.copy()
        vis2 = img2.copy()

        idx = np.random.choice(n, 3, replace=False)

        for i, k in enumerate(idx):
            color = colors[i]
            x1, y1, x2, y2 = M[k]

            p = np.array([x1, y1, 1.0], dtype=np.float64)
            q = np.array([x2, y2, 1.0], dtype=np.float64)

            l = F @ p      # line in img2
            m = F.T @ q    # line in img1

            cv2.circle(vis1, (int(round(x1)), int(round(y1))), 6, color, -1)
            cv2.circle(vis2, (int(round(x2)), int(round(y2))), 6, color, -1)

            seg2 = line_endpoints_in_image(l, w2, h2)
            if seg2 is not None:
                cv2.line(vis2, seg2[0], seg2[1], color, 2)

            seg1 = line_endpoints_in_image(m, w1, h1)
            if seg1 is not None:
                cv2.line(vis1, seg1[0], seg1[1], color, 2)

        show = np.concatenate([vis1, vis2], axis=1)
        cv2.imshow(win_name, show)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            cv2.destroyWindow(win_name)
            break

if __name__ == '__main__':

    # setup for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    temple1 = cv2.imread('./A3_P1_Data/temple1.png')
    temple1 = np.asarray(temple1, dtype=np.float32)
    temple2 = cv2.imread('./A3_P1_Data/temple2.png')
    temple2 = np.asarray(temple2, dtype=np.float32)

    house1 = cv2.imread('./A3_P1_Data/house1.jpg')
    house1 = np.asarray(house1, dtype=np.float32)
    house2 = cv2.imread('./A3_P1_Data/house2.jpg')
    house2 = np.asarray(house2, dtype=np.float32)

    library1 = cv2.imread('./A3_P1_Data/library1.jpg')
    library1 = np.asarray(library1, dtype=np.float32)
    library2 = cv2.imread('./A3_P1_Data/library2.jpg')
    library2 = np.asarray(library2, dtype=np.float32)

    M_temple = np.loadtxt('./A3_P1_Data/temple_matches.txt')
    set_image_sizes(temple1.shape[1], temple1.shape[0], temple2.shape[1], temple2.shape[0])
    _, _, temple_F = show_error(M_temple, 'temple', th=0.9)


    M_house = np.loadtxt('./A3_P1_Data/house_matches.txt')
    set_image_sizes(house1.shape[1], house1.shape[0], house2.shape[1], house2.shape[0])
    _, _, house_F = show_error(M_house, 'house', th=0.9,  format='jpg')

    
    M_library = np.loadtxt('./A3_P1_Data/library_matches.txt')
    set_image_sizes(library1.shape[1], library1.shape[0], library2.shape[1], library2.shape[0])
    _, _, library_F = show_error(M_library, 'library', th=0.9, format='jpg')

    draw_epipolar_demo(temple1.astype(np.uint8), temple2.astype(np.uint8), M_temple, temple_F, win_name='Temple Epipolar')
    draw_epipolar_demo(house1.astype(np.uint8), house2.astype(np.uint8), M_house, house_F, win_name='House Epipolar')
    draw_epipolar_demo(library1.astype(np.uint8), library2.astype(np.uint8), M_library, library_F, win_name='Library Epipolar')