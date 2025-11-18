import cv2
import numpy as np
import time

def get_kp_des(img: np.ndarray):
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return kp, des

def hamming_distance(d1: np.ndarray, d2: np.ndarray) -> int:
    # des.dtype = uint8
    # des.shape = (32,)
    x = np.bitwise_xor(d1, d2)
    return int(np.sum(np.unpackbits(x)))
    

def match_descriptors(des1: np.ndarray, des2: np.ndarray):
     
    matches = []
    n1 = des1.shape[0]
    n2 = des2.shape[0]
    for i in range(n1):
        distances = []
        for j in range(n2):
            dist = hamming_distance(des1[i], des2[j])
            distances.append((dist, j))
        
        if len(distances) < 2:
            continue

        distances.sort(key=lambda x: x[0])
        best_match = distances[0]
        second_best_match = distances[1]

        best_dist = best_match[0]
        second_best_dist = second_best_match[0]
        ratio = best_dist / (second_best_dist + 1e-6)

        if ratio < 0.75:
            matches.append((i, best_match[1], best_dist))

    matches.sort(key=lambda x: x[2])
    return matches
    

def visualize_matches(img1, img2, kp1, kp2, matches):
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, 
                                [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _imgIdx=0, _distance=m[2]) for m in matches],
                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', match_img)
    cv2.waitKey(0)


def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    max_dist = np.max(np.sqrt(np.sum(centered**2, axis=1)))
    scale = np.sqrt(2) / (max_dist + 1e-6)
    
    T = np.array(
        [[scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1]],
    dtype=np.float32
    )

    N = pts.shape[0]
    homogeneous_pts = np.hstack((pts, np.ones((N, 1), dtype=np.float32)))
    normalized_pts = (np.dot(T, homogeneous_pts.T)).T  # (3, 3) X (3, N)
    heterogeneous_pts = normalized_pts[:, :2] / normalized_pts[:, 2:3]  # (N, 2)
    return heterogeneous_pts, T


def compute_homography(srcP, destP):
    # srcP, destP: (N, 2)
    # N = # of matched feature points
    # 2 = (x, y)

    N = srcP.shape[0]

    norm_srcP, T_srcP = normalize_points(srcP)
    norm_destP, T_destP = normalize_points(destP)

    A = []
    for i in range(N):
        x, y = norm_srcP[i]
        x_p, y_p = norm_destP[i]

        A.append([-x, -y, -1, 0, 0, 0, x * x_p, y * x_p, x_p])
        A.append([0, 0, 0, -x, -y, -1, x * y_p, y * y_p, y_p])

    A = np.array(A, dtype=np.float32)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)

    H_final = np.linalg.inv(T_destP) @ H @ T_srcP
    H_final = H_final / (H_final[2, 2] + 1e-6)  # normalize
    return H_final


def compute_homography_ransac(srcP, destP, th):
    start_time = time.time()
    best_inlier = None

    while time.time() - start_time < 2.9:
        idx = np.random.choice(srcP.shape[0], 4, replace=False)
        src_sample = srcP[idx]
        dest_sample = destP[idx]

        H = compute_homography(src_sample, dest_sample)
        homogeneous_srcP = np.hstack((srcP, np.ones((srcP.shape[0], 1), dtype=np.float32)))
        projected = (np.dot(H, homogeneous_srcP.T)).T  # (N, 3)
        projected = projected[:, :2] / (projected[:, 2:3] + 1e-6)  # normalize

        error = np.sqrt(np.sum((projected - destP)**2, axis=1))
        inlier = np.where(error < th)[0]

        if best_inlier is None or len(inlier) > len(best_inlier):
            best_inlier = inlier

    final_src = srcP[best_inlier]
    final_dst = destP[best_inlier]
    H_final = compute_homography(final_src, final_dst)
    return H_final

def visualize_warp_composed(warped_img, composed_img, left_title, right_title, win_name):
    left = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(composed_img, cv2.COLOR_GRAY2BGR)

    cv2.putText(left, left_title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    cv2.putText(right, right_title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    
    concat = np.hstack((left, right))
    cv2.imshow(win_name, concat)
    cv2.waitKey(0)


if __name__ == "__main__":
    import random
    np.random.seed(43)
    random.seed(43)

    cover = cv2.imread('./A2_Images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    desk = cv2.imread('./A2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)
    hp_cover = cv2.imread('./A2_Images/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
    

    cover_kp, cover_des = get_kp_des(cover)
    desk_kp, desk_des = get_kp_des(desk)

    # 2-1
    matches = match_descriptors(desk_des, cover_des)
    if len(matches) < 15:
        raise ValueError(f"The number of matches must be at least 15. But got {len(matches)}")
    visualize_matches(desk, cover, desk_kp, cover_kp, matches[:10])

    # 2-2
    src_pts = np.float32([cover_kp[m[1]].pt for m in matches])
    dst_pts = np.float32([desk_kp[m[0]].pt  for m in matches])
    desk_h, desk_w = desk.shape

    # 2-4-b Homography with Normalization
    H_norm = compute_homography(src_pts, dst_pts)
    warped_norm = cv2.warpPerspective(cover, H_norm, (desk_w, desk_h))

    composed_norm = desk.copy()
    mask_norm = warped_norm > 0
    composed_norm[mask_norm] = warped_norm[mask_norm]
    visualize_warp_composed(warped_norm,
                            composed_norm,
                            "Warped Cover (Normalization)",
                            "Composite (Normalization)",
                            "Homography with Normalization"
                            )


    # 2-4-b Ransac Homography
    print("Computing Homography with RANSAC...")
    start = time.time()
    H_ransac = compute_homography_ransac(src_pts, dst_pts, th=3)
    print(f"RANSAC took {time.time() - start:.2f} seconds")

    warped_ransac = cv2.warpPerspective(cover, H_ransac, (desk_w, desk_h))
    composed_ransac = desk.copy()
    mask_ransac = warped_ransac > 0
    composed_ransac[mask_ransac] = warped_ransac[mask_ransac]

    visualize_warp_composed(warped_ransac,
                            composed_ransac,
                            "Warped Cover (RANSAC)",
                            "Composite (RANSAC)",
                            "Homography with RANSAC"
                            )

    # 2-4-c Apply Homography to HP Cover Image
    hp_cover_resized = cv2.resize(hp_cover, (cover.shape[1], cover.shape[0]))
    warped_hp = cv2.warpPerspective(hp_cover_resized, H_ransac, (desk_w, desk_h))
    composed_hp = desk.copy()
    mask_hp = warped_hp > 0
    composed_hp[mask_hp] = warped_hp[mask_hp]
    visualize_warp_composed(warped_hp,
                            composed_hp,
                            "Warped HP Cover (RANSAC)",
                            "Composite HP Cover (RANSAC)",
                            "Homography with RANSAC"
                            )