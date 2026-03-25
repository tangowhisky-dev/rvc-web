"""Standalone FAISS index builder — run as a subprocess to isolate native memory.

Usage: python _index_worker.py <project_root>

project_root is the rvc-web/ directory (NOT the old RVC submodule root).
The feature directory is at project_root/logs/rvc_finetune_active/3_feature768/.

Prints the path to the added_IVF*.index file on stdout (last line).
Exits 0 on success, 1 on failure (error on stderr).
"""

import multiprocessing
import os
import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: _index_worker.py <project_root>", file=sys.stderr)
        sys.exit(1)

    project_root = sys.argv[1]
    exp_name = "rvc_finetune_active"
    exp_dir = os.path.join(project_root, "logs", exp_name)
    feature_dir = os.path.join(exp_dir, "3_feature768")

    try:
        import faiss
        import numpy as np

        if not os.path.isdir(feature_dir):
            raise RuntimeError(f"Feature directory not found: {feature_dir}")

        npy_files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".npy"))
        if not npy_files:
            raise RuntimeError(f"No .npy files found in {feature_dir}")

        npys = [np.load(os.path.join(feature_dir, fname)) for fname in npy_files]
        big_npy = np.concatenate(npys, axis=0)

        # Shuffle
        idx = np.arange(big_npy.shape[0])
        np.random.shuffle(idx)
        big_npy = big_npy[idx]

        # Reduce with MiniBatchKMeans only for large feature sets
        if big_npy.shape[0] > 200_000:
            from sklearn.cluster import MiniBatchKMeans
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10_000,
                    verbose=True,
                    batch_size=256 * multiprocessing.cpu_count(),
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )

        np.save(os.path.join(exp_dir, "total_fea.npy"), big_npy)

        n_ivf = min(int(16 * (big_npy.shape[0] ** 0.5)), big_npy.shape[0] // 39)
        n_ivf = max(n_ivf, 1)

        index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1
        index.train(big_npy)

        trained_path = os.path.join(
            exp_dir,
            f"trained_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_v2.index",
        )
        faiss.write_index(index, trained_path)

        # Add vectors in batches and save the searchable index
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])

        added_path = os.path.join(
            exp_dir,
            f"added_IVF{n_ivf}_Flat_nprobe_{index_ivf.nprobe}_{exp_name}_v2.index",
        )
        faiss.write_index(index, added_path)

        # Print result path — caller reads last stdout line
        print(added_path, flush=True)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
