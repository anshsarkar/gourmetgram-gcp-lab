import os
import sys
import glob
import random
import time
import base64
import uuid
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

SERVICE_URL = os.environ.get("SERVICE_URL", "").rstrip("/")
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.expanduser("~/Food-11"))
GCS_UPLOAD_BUCKET = os.environ.get("GCS_UPLOAD_BUCKET", "")

# Burst config
NUM_BURSTS = 3
IMAGES_PER_BURST = 200
FINAL_BURST_MULTIPLIER = 4  # last burst sends 4x images concurrently
CONCURRENT_WORKERS = 20
PAUSE_BETWEEN_BURSTS = 60  # seconds


def get_image_paths(dataset_dir):
    """Get image paths from the evaluation set only — simulates unseen production data."""
    paths = []
    eval_dir = os.path.join(dataset_dir, "evaluation")
    if os.path.isdir(eval_dir):
        paths.extend(glob.glob(os.path.join(eval_dir, "**", "*.jpg"), recursive=True))
    if not paths:
        # try flat structure (images directly in eval dir)
        if os.path.isdir(eval_dir):
            for f in os.listdir(eval_dir):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(eval_dir, f))
    return paths


def send_predict_request(image_path):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
    resp = requests.post(
        f"{SERVICE_URL}/api/predict",
        json={"image": img_b64},
        timeout=60,
    )
    return resp.status_code, resp.json()


def send_load_request():
    resp = requests.get(f"{SERVICE_URL}/test", timeout=60)
    return resp.status_code, resp.text


def upload_to_gcs(image_path):
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(GCS_UPLOAD_BUCKET)
    filename = f"{uuid.uuid4().hex}_{os.path.basename(image_path)}"
    blob = bucket.blob(f"uploads/{filename}")
    blob.upload_from_filename(image_path, content_type="image/jpeg")
    return filename


def run_burst_sequential(mode, sample, burst_size):
    errors = 0
    for i in range(burst_size):
        try:
            if mode == "predict":
                status, result = send_predict_request(sample[i])
                if status == 200:
                    print(f"  [{i+1}/{burst_size}] {os.path.basename(sample[i])} -> {result['prediction']} ({result['confidence']:.2f})")
                else:
                    print(f"  [{i+1}/{burst_size}] {os.path.basename(sample[i])} -> HTTP {status}")
                    errors += 1
            elif mode == "upload":
                filename = upload_to_gcs(sample[i])
                print(f"  [{i+1}/{burst_size}] {os.path.basename(sample[i])} -> gs://{GCS_UPLOAD_BUCKET}/uploads/{filename}")
            else:
                status, result = send_load_request()
                if status == 200:
                    print(f"  [{i+1}/{burst_size}] /test -> {result}")
                else:
                    print(f"  [{i+1}/{burst_size}] /test -> HTTP {status}")
                    errors += 1
        except Exception as e:
            print(f"  [{i+1}/{burst_size}] ERROR: {e}")
            errors += 1
    return errors


def run_burst_concurrent(mode, sample, burst_size):
    errors = 0
    completed = 0

    def do_request(idx):
        if mode == "predict":
            return idx, send_predict_request(sample[idx])
        elif mode == "upload":
            return idx, (200, upload_to_gcs(sample[idx]))
        else:
            return idx, send_load_request()

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {executor.submit(do_request, i): i for i in range(burst_size)}
        for future in as_completed(futures):
            completed += 1
            try:
                idx, result = future.result()
                if mode == "predict":
                    status, data = result
                    if status == 200:
                        print(f"  [{completed}/{burst_size}] {os.path.basename(sample[idx])} -> {data['prediction']} ({data['confidence']:.2f})")
                    else:
                        print(f"  [{completed}/{burst_size}] {os.path.basename(sample[idx])} -> HTTP {status}")
                        errors += 1
                elif mode == "upload":
                    _, filename = result
                    print(f"  [{completed}/{burst_size}] {os.path.basename(sample[idx])} -> uploaded")
                else:
                    status, data = result
                    if status == 200:
                        print(f"  [{completed}/{burst_size}] /test -> {data}")
                    else:
                        print(f"  [{completed}/{burst_size}] /test -> HTTP {status}")
                        errors += 1
            except Exception as e:
                print(f"  [{completed}/{burst_size}] ERROR: {e}")
                errors += 1
    return errors


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

    final_burst_size = IMAGES_PER_BURST * FINAL_BURST_MULTIPLIER
    total_requests = IMAGES_PER_BURST * (NUM_BURSTS - 1) + final_burst_size

    print(f"Mode: {args.mode}")
    if args.mode == "upload":
        print(f"Target bucket: gs://{GCS_UPLOAD_BUCKET}/uploads/")
    else:
        print(f"Target: {SERVICE_URL}")
    print(f"Plan: {NUM_BURSTS - 1} bursts x {IMAGES_PER_BURST} sequential, "
          f"then 1 burst x {final_burst_size} concurrent ({CONCURRENT_WORKERS} workers)")
    print(f"Total: {total_requests} requests, {PAUSE_BETWEEN_BURSTS}s pause between bursts")
    print()

    total_sent = 0
    total_errors = 0

    for burst in range(NUM_BURSTS):
        is_final = (burst == NUM_BURSTS - 1)
        burst_size = final_burst_size if is_final else IMAGES_PER_BURST

        if is_final:
            print(f"--- Burst {burst + 1}/{NUM_BURSTS} (FINAL — {burst_size} concurrent requests) ---")
        else:
            print(f"--- Burst {burst + 1}/{NUM_BURSTS} ---")

        burst_start = time.time()

        sample = None
        if args.mode in ("predict", "upload"):
            sample = random.choices(image_paths, k=burst_size)

        if is_final:
            burst_errors = run_burst_concurrent(args.mode, sample, burst_size)
        else:
            burst_errors = run_burst_sequential(args.mode, sample, burst_size)

        elapsed = time.time() - burst_start
        total_sent += burst_size
        total_errors += burst_errors
        print(f"  Burst done: {burst_size - burst_errors}/{burst_size} OK in {elapsed:.1f}s")

        if burst < NUM_BURSTS - 1:
            print(f"  Pausing {PAUSE_BETWEEN_BURSTS}s...")
            time.sleep(PAUSE_BETWEEN_BURSTS)

    print()
    print(f"Done. Sent {total_sent} requests, {total_errors} errors.")


if __name__ == "__main__":
    main()
