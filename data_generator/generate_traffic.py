import os
import sys
import glob
import random
import time
import base64
import uuid
import argparse
import requests

SERVICE_URL = os.environ.get("SERVICE_URL", "").rstrip("/")
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.expanduser("~/Food-11"))
GCS_UPLOAD_BUCKET = os.environ.get("GCS_UPLOAD_BUCKET", "")

# Burst config
NUM_BURSTS = 3
IMAGES_PER_BURST = 200
PAUSE_BETWEEN_BURSTS = 60  # seconds


def get_image_paths(dataset_dir):
    paths = []
    for split in ["training", "validation", "evaluation"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            continue
        paths.extend(glob.glob(os.path.join(split_dir, "**", "*.jpg"), recursive=True))
    if not paths:
        # try flat structure (images directly in split dirs)
        for split in ["training", "validation", "evaluation"]:
            split_dir = os.path.join(dataset_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for f in os.listdir(split_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(split_dir, f))
    return paths


def send_predict_request(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(
        f"{SERVICE_URL}/api/predict",
        json={"image": img_b64},
        timeout=30,
    )
    return resp.status_code, resp.json()


def send_load_request():
    resp = requests.get(f"{SERVICE_URL}/test", timeout=30)
    return resp.status_code, resp.text


def upload_to_gcs(image_path):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_UPLOAD_BUCKET)
    filename = f"{uuid.uuid4().hex}_{os.path.basename(image_path)}"
    blob = bucket.blob(f"uploads/{filename}")
    blob.upload_from_filename(image_path, content_type="image/jpeg")
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["predict", "load", "upload"], default="predict",
        help="predict: send images to /api/predict (collects data to GCS). "
             "load: hit /test endpoint to generate CPU load for scaling demos. "
             "upload: upload images to GCS bucket for Eventarc triggers."
    )
    args = parser.parse_args()

    if args.mode in ("predict", "load") and not SERVICE_URL:
        print("Error: set SERVICE_URL environment variable (e.g. https://your-cloud-run-url)")
        sys.exit(1)

    if args.mode == "upload" and not GCS_UPLOAD_BUCKET:
        print("Error: set GCS_UPLOAD_BUCKET environment variable (e.g. your-net-id-eventarc-bucket)")
        sys.exit(1)

    image_paths = None
    if args.mode in ("predict", "upload"):
        image_paths = get_image_paths(DATASET_DIR)
        if not image_paths:
            print(f"Error: no images found in {DATASET_DIR}")
            sys.exit(1)
        print(f"Found {len(image_paths)} images in {DATASET_DIR}")

    print(f"Mode: {args.mode}")
    if args.mode == "upload":
        print(f"Target bucket: gs://{GCS_UPLOAD_BUCKET}/uploads/")
    else:
        print(f"Target: {SERVICE_URL}")
    print(f"Plan: {NUM_BURSTS} bursts x {IMAGES_PER_BURST} requests, {PAUSE_BETWEEN_BURSTS}s pause between bursts")
    print()

    total_sent = 0
    total_errors = 0

    for burst in range(NUM_BURSTS):
        print(f"--- Burst {burst + 1}/{NUM_BURSTS} ---")
        burst_errors = 0
        burst_start = time.time()

        if args.mode in ("predict", "upload"):
            sample = random.choices(image_paths, k=IMAGES_PER_BURST)

        for i in range(IMAGES_PER_BURST):
            try:
                if args.mode == "predict":
                    status, result = send_predict_request(sample[i])
                    if status == 200:
                        print(f"  [{i+1}/{IMAGES_PER_BURST}] {os.path.basename(sample[i])} -> {result['prediction']} ({result['confidence']:.2f})")
                    else:
                        print(f"  [{i+1}/{IMAGES_PER_BURST}] {os.path.basename(sample[i])} -> HTTP {status}")
                        burst_errors += 1
                elif args.mode == "upload":
                    filename = upload_to_gcs(sample[i])
                    print(f"  [{i+1}/{IMAGES_PER_BURST}] {os.path.basename(sample[i])} -> gs://{GCS_UPLOAD_BUCKET}/uploads/{filename}")
                else:
                    status, result = send_load_request()
                    if status == 200:
                        print(f"  [{i+1}/{IMAGES_PER_BURST}] /test -> {result}")
                    else:
                        print(f"  [{i+1}/{IMAGES_PER_BURST}] /test -> HTTP {status}")
                        burst_errors += 1
            except Exception as e:
                print(f"  [{i+1}/{IMAGES_PER_BURST}] ERROR: {e}")
                burst_errors += 1

        elapsed = time.time() - burst_start
        total_sent += IMAGES_PER_BURST
        total_errors += burst_errors
        print(f"  Burst done: {IMAGES_PER_BURST - burst_errors}/{IMAGES_PER_BURST} OK in {elapsed:.1f}s")

        if burst < NUM_BURSTS - 1:
            print(f"  Pausing {PAUSE_BETWEEN_BURSTS}s...")
            time.sleep(PAUSE_BETWEEN_BURSTS)

    print()
    print(f"Done. Sent {total_sent} requests, {total_errors} errors.")


if __name__ == "__main__":
    main()
