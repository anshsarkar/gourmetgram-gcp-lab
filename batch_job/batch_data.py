import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from google.cloud import storage

MAX_WORKERS = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_next_version(training_bucket):
    """Determine the next version number by scanning existing version folders."""
    logger.info("Scanning training bucket for existing versions...")
    blobs = training_bucket.list_blobs(prefix="datasets/Food-11/v", delimiter="/")
    # Force iteration to populate prefixes
    list(blobs)

    versions = []
    for prefix in blobs.prefixes:
        # prefix looks like: datasets/Food-11/v3/
        folder_name = prefix.rstrip("/").split("/")[-1]
        if folder_name.startswith("v") and folder_name[1:].isdigit():
            versions.append(int(folder_name[1:]))

    if versions:
        logger.info(f"Found existing versions: {sorted(versions)}")
    else:
        logger.info("No existing versions found. Starting from v1.")

    if not versions:
        return 1
    return max(versions) + 1


def batch_copy():
    staging_bucket_name = os.environ.get("GCS_STAGING_BUCKET")
    training_bucket_name = os.environ.get("GCS_TRAINING_BUCKET")

    if not staging_bucket_name or not training_bucket_name:
        logger.error("GCS_STAGING_BUCKET and GCS_TRAINING_BUCKET must be set")
        sys.exit(1)

    logger.info(f"Staging bucket:  gs://{staging_bucket_name}")
    logger.info(f"Training bucket: gs://{training_bucket_name}")

    client = storage.Client()
    staging_bucket = client.bucket(staging_bucket_name)
    training_bucket = client.bucket(training_bucket_name)

    # List all image blobs under incoming/
    logger.info("Listing images in incoming/...")
    blobs = list(staging_bucket.list_blobs(prefix="incoming/"))
    image_blobs = [
        b for b in blobs
        if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_blobs:
        logger.info("No new images in incoming/. Skipping — no version created.")
        return

    version = get_next_version(training_bucket)
    logger.info(f"Found {len(image_blobs)} new images. Creating v{version}...")

    # Copy images to versioned training folder (parallel)
    copied = 0
    class_counts = {}
    lock = threading.Lock()
    total = len(image_blobs)

    def copy_blob(blob):
        nonlocal copied
        relative_path = blob.name[len("incoming/"):]
        dest_path = f"datasets/Food-11/v{version}/training/{relative_path}"
        for attempt in range(3):
            try:
                staging_bucket.copy_blob(blob, training_bucket, new_name=dest_path)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    logger.warning(f"Retry {attempt+1} for {blob.name}: {e}")
                else:
                    raise
        class_name = relative_path.split("/")[0]
        with lock:
            copied += 1
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if copied % 50 == 0:
                logger.info(f"  Progress: {copied}/{total} images copied...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(copy_blob, blob) for blob in image_blobs]
        for future in as_completed(futures):
            future.result()  # raises if a copy failed

    logger.info(f"Copy complete. {copied} images copied to v{version}.")

    # Write metadata for this version
    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_images": copied,
        "class_counts": dict(sorted(class_counts.items())),
    }
    metadata_blob = training_bucket.blob(
        f"datasets/Food-11/v{version}/metadata.json"
    )
    metadata_blob.upload_from_string(
        json.dumps(metadata, indent=2), content_type="application/json"
    )
    logger.info(f"Metadata written to datasets/Food-11/v{version}/metadata.json")

    # Delete incoming/ data from staging bucket after successful copy (parallel)
    all_blobs_to_delete = blobs  # includes images + directory markers
    logger.info(f"Cleaning up {len(all_blobs_to_delete)} objects from staging bucket...")
    deleted = 0

    def delete_blob(blob):
        nonlocal deleted
        blob.delete()
        with lock:
            deleted += 1
            if deleted % 50 == 0:
                logger.info(f"  Cleanup progress: {deleted}/{len(all_blobs_to_delete)} deleted...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(delete_blob, blob) for blob in all_blobs_to_delete]
        for future in as_completed(futures):
            future.result()

    logger.info("Staging bucket cleanup complete.")

    logger.info(f"Done. v{version} created with {copied} images in "
                f"gs://{training_bucket_name}/datasets/Food-11/v{version}/")
    logger.info(f"Per-class counts:\n{json.dumps(class_counts, indent=2)}")


if __name__ == "__main__":
    logger.info("=== Batch Data Job started ===")
    batch_copy()
    logger.info("=== Batch Data Job finished ===")
