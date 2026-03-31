import json
import os
import sys
from datetime import datetime, timezone

from google.cloud import storage


def get_next_version(training_bucket):
    blobs = training_bucket.list_blobs(prefix="datasets/Food-11/v", delimiter="/")
    # Force iteration to populate prefixes
    list(blobs)

    versions = []
    for prefix in blobs.prefixes:
        # prefix looks like: datasets/Food-11/v3/
        folder_name = prefix.rstrip("/").split("/")[-1]
        if folder_name.startswith("v") and folder_name[1:].isdigit():
            versions.append(int(folder_name[1:]))

    if not versions:
        return 1
    return max(versions) + 1


def batch_copy():
    staging_bucket_name = os.environ.get("GCS_STAGING_BUCKET")
    training_bucket_name = os.environ.get("GCS_TRAINING_BUCKET")

    if not staging_bucket_name or not training_bucket_name:
        print("ERROR: GCS_STAGING_BUCKET and GCS_TRAINING_BUCKET must be set")
        sys.exit(1)

    client = storage.Client()
    staging_bucket = client.bucket(staging_bucket_name)
    training_bucket = client.bucket(training_bucket_name)

    # List all image blobs under incoming/
    blobs = list(staging_bucket.list_blobs(prefix="incoming/"))
    image_blobs = [
        b for b in blobs
        if b.name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_blobs:
        print("No new images in incoming/. Skipping — no version created.")
        return

    version = get_next_version(training_bucket)
    print(f"Found {len(image_blobs)} new images. Creating v{version}...")

    # Copy images to versioned training folder
    copied = 0
    class_counts = {}
    for blob in image_blobs:
        # blob.name: incoming/class_03/abc123.jpg
        # dest:      datasets/Food-11/v{N}/training/class_03/abc123.jpg
        relative_path = blob.name[len("incoming/"):]  # class_03/abc123.jpg
        dest_path = f"datasets/Food-11/v{version}/training/{relative_path}"

        staging_bucket.copy_blob(blob, training_bucket, new_name=dest_path)
        copied += 1

        # Track per-class counts
        class_name = relative_path.split("/")[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if copied % 50 == 0:
            print(f"  Copied {copied}/{len(image_blobs)}...")

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

    # Delete incoming/ data from staging bucket after successful copy
    print("Cleaning up staging bucket...")
    for blob in image_blobs:
        blob.delete()
    # Also delete any leftover empty "directory" markers
    for blob in blobs:
        if not blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
            blob.delete()

    print(f"Done. v{version} created with {copied} images in "
          f"gs://{training_bucket_name}/datasets/Food-11/v{version}/")
    print(f"Per-class counts: {json.dumps(class_counts, indent=2)}")


if __name__ == "__main__":
    batch_copy()
