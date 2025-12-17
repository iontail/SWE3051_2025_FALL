import numpy as np
from glob import glob
from tqdm import tqdm


def load_sift_file(path: str):
    return np.fromfile(path, dtype=np.uint8).reshape(-1, 128).astype(np.float32)

def load_all_sift(dir_path: str):
    sift_list = []
    for p in tqdm(sorted(glob(f"{dir_path}/*.sift"))):
        sift_list.append(load_sift_file(p))
    return np.vstack(sift_list)

def load_cnn_file(path: str):
    return np.fromfile(path, dtype=np.float32).reshape(14, 14, 512)

def load_all_cnn(dir_path: str):
    cnn_list = []
    for p in tqdm(sorted(glob(f"{dir_path}/*.cnn"))):
        cnn_list.append(load_cnn_file(p))
    return np.vstack(cnn_list)


if __name__ == "__main__":
    sift_features = load_all_sift('./features/sift')
    sift_feature = load_sift_file('./features/sift/0000.sift')
    print(f"SIFT features shape: {sift_features.shape}")
    print(f"SIFT feature 0000.sift shape: {sift_feature.shape}")


    cnn_features = load_all_cnn('./features/cnn')
    cnn_feature = load_cnn_file('./features/cnn/0000.cnn')
    print(f"CNN features shape: {cnn_features.shape}")
    print(f"CNN feature 0000.cnn shape: {cnn_feature.shape}")