import struct
import random
import numpy as np
from tqdm import tqdm

from utils import load_sift_file, load_cnn_file
from vlad import VLAD

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)

def compute_gap(cnn_feat: np.ndarray, max_gap: bool = False) -> np.ndarray:

    if not max_gap:
        gap = cnn_feat.mean(axis=(0, 1)).astype(np.float32)  # (512,)
    else:
        # 7x7 max pooling with stride 7
        # (14,14,512) -> (2,2,512)
        pooled = np.zeros((2, 2, cnn_feat.shape[2]), dtype=np.float32)

        for i in range(2):
            for j in range(2):
                h_start = i * 7
                h_end = h_start + 7
                w_start = j * 7
                w_end = w_start + 7

                pooled[i, j] = np.max(
                    cnn_feat[h_start:h_end, w_start:w_end],
                    axis=(0, 1)
                )

        # GAP on pooled feature map
        gap = pooled.mean(axis=(0, 1))  # (512,)

    return l2_normalize(gap).astype(np.float32)

def write_descriptors(path: str, descs: np.ndarray):
    N, D = descs.shape
    with open(path, "wb") as f:
        f.write(struct.pack("ii", int(N), int(D)))  # int32 N, int32 D
        f.write(descs.astype(np.float32).tobytes(order="C"))

if __name__ == "__main__":
    N = 2000
    K = 28
    D_CNN = 512
    D_VLAD = K * 128 # 3584
    D = D_CNN + D_VLAD  # 4096
    MAX_GAP = True
    TRAIN = False # Please set TRAIN = False when you test

    
    centers_path = f"./kmeans_centers_K{K}.npy"
    out_path = "A3_2021312134_maxgap.des"

    # setup for reproducibility
    np.random.seed(42)
    random.seed(42)

    vlad = VLAD(k=K, seed=42)

    if TRAIN:
        vlad.fit(n_images=N, sift_path="./features/sift")
        centers_path = vlad.save(centers_path)
    else:
        vlad.load(centers_path)

    descs = np.zeros((N, D), dtype=np.float32)
    for i in tqdm(range(N)):
        # cnn GAP (512)
        cnn = load_cnn_file(f"./features/cnn/{i:04d}.cnn")  # (14,14,512)
        cnn_desc = compute_gap(cnn, max_gap=MAX_GAP)                         # (512,)

        # sift VLAD (3584)
        sift = load_sift_file(f"./features/sift/{i:04d}.sift")  # (n,128)
        vlad_desc = vlad.predict(sift)                           # (3584,)

        # concat
        desc = np.concatenate([cnn_desc, vlad_desc], axis=0).astype(np.float32)
        desc = l2_normalize(desc).astype(np.float32) # normalize as concatenating cnn and vlad
        descs[i] = desc

    write_descriptors(out_path, descs)
    print(f"Saved: {out_path} | shape={descs.shape} | dtype={descs.dtype}")