"""
Fetch the SECOM dataset from the UCI ML repository and verify it against
known SHA-256 hashes so that downstream notebooks/training run on
exactly the data this project was developed against.

Usage:
    python scripts/fetch_data.py [--out-dir data/raw]

The script is idempotent: if both files already exist with matching
hashes it skips the download.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fetch_data")

# Pinned upstream artifacts. If UCI ever rotates these files, both the
# URL and the hash need updating in lockstep.
SECOM_FILES = {
    "secom.data": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
        "sha256": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    },
    "secom_labels.data": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data",
        "sha256": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
    },
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_one(url: str, dest: str, expected_sha: str) -> None:
    if os.path.exists(dest):
        actual = sha256_of(dest)
        if actual == expected_sha:
            logger.info("skip: %s already exists with matching hash", dest)
            return
        logger.warning("hash mismatch on existing %s; re-downloading", dest)

    logger.info("download: %s -> %s", url, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310

    actual = sha256_of(dest)
    if actual != expected_sha:
        os.remove(dest)
        raise RuntimeError(
            f"SHA-256 mismatch for {dest}\n  expected: {expected_sha}\n  got:      {actual}"
        )
    logger.info("verified: %s (sha256 ok)", dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="data/raw",
        help="Directory to write the raw files to (default: data/raw).",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for filename, meta in SECOM_FILES.items():
        dest = os.path.join(args.out_dir, filename)
        try:
            fetch_one(meta["url"], dest, meta["sha256"])
        except Exception as e:  # noqa: BLE001
            logger.error("failed: %s -- %s", filename, e)
            return 1

    logger.info("\nAll SECOM files in %s ready.", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
