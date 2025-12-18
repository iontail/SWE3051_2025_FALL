import numpy as np
from sklearn.cluster import KMeans
from utils import load_sift_file


class VLAD:
    def __init__(self, k: int = 28, seed: int = 42):
        self.k = k
        self.seed = seed
        self.centers = None # (K,128)

    def fit(self, n_images: int = 2000, sift_path: str = "./features/sift"):
        feats = []
        for i in range(n_images):
            sift = load_sift_file(f"{sift_path}/{i:04d}.sift")
            feats.append(sift)

        X = np.vstack(feats) # (total_SIFT, 128)

        kmeans = KMeans(
            n_clusters=self.k,
            random_state=self.seed,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_.astype(np.float32)
        return self

    def save(self, path: str = None):
        if path is None:
            path = f"kmeans_centers_K{self.k}.npy"
        np.save(path, self.centers.astype(np.float32))
        return path

    def load(self, path: str = None):
        if path is None:
            path = f"kmeans_centers_K{self.k}.npy"
        self.centers = np.load(path).astype(np.float32)
        return self

    def predict(self, sift: np.ndarray):
        centers = self.centers
        K, D = centers.shape

        x2 = np.sum(sift * sift, axis=1, keepdims=True)          # (n,1)
        c2 = np.sum(centers * centers, axis=1, keepdims=True).T  # (1,K)
        l2_dist = x2 + c2 - 2.0 * (sift @ centers.T)             # (n,K)
        labels = np.argmin(l2_dist, axis=1)                      # (n,)

        vlad = np.zeros((K, D), dtype=np.float32)
        for kk in range(K):
            idx = (labels == kk)
            if np.any(idx):
                vlad[kk] = np.sum(sift[idx] - centers[kk], axis=0)

        # intra-normalization
        for kk in range(K):
            vlad[kk] /= (np.linalg.norm(vlad[kk]) + 1e-9) # residual sum

        vlad = vlad.reshape(-1)
        vlad /= (np.linalg.norm(vlad) + 1e-12)
        return vlad.astype(np.float32)


if __name__ == "__main__":
    vlad = VLAD(k=28, seed=42)
    
    vlad.fit(n_images=2000, sift_path="./features/sift")
    centers_path = vlad.save() # "kmeans_centers_K28.npy"
    print("saved:", centers_path)

    vlad.load(centers_path)
    for i in [0, 1, 2, 3, 4]:
        sift = load_sift_file(f"./features/sift/{i:04d}.sift")
        vec = vlad.predict(sift)
        print(f"[{i:04d}] vlad: {vec.shape}, ||v||2={np.linalg.norm(vec):.6f}")