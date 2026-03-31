import argparse

from google.cloud import aiplatform
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component
from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output


# ---------------------------------------------------------------------------
# Component 1: Prepare Data
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-storage"],
)
def prepare_data(
    training_bucket: str,
    collated_dataset: Output[Dataset],
):
    """Collate untrained dataset versions and split into train/val (80/20)."""
    import json
    import logging
    import random
    import traceback
    from google.cloud import storage

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"=== prepare_data started. Bucket: {training_bucket} ===")

        client = storage.Client()
        bucket = client.bucket(training_bucket)

        # Read training metadata to find last trained version
        metadata_blob = bucket.blob("training_metadata.json")
        if metadata_blob.exists():
            metadata = json.loads(metadata_blob.download_as_text())
            last_trained = metadata.get("last_trained_version", 0)
            logger.info(f"Found training metadata. Last trained version: v{last_trained}")
        else:
            metadata = {}
            last_trained = 0
            logger.info("No training metadata found. Will process all versions.")

        # Find all dataset versions
        blobs = bucket.list_blobs(prefix="datasets/Food-11/v", delimiter="/")
        list(blobs)
        versions = []
        for prefix in blobs.prefixes:
            folder_name = prefix.rstrip("/").split("/")[-1]
            if folder_name.startswith("v") and folder_name[1:].isdigit():
                v = int(folder_name[1:])
                if v > last_trained:
                    versions.append(v)

        if not versions:
            logger.info("No new versions to process. Exiting.")
            collated_dataset.metadata["versions_collated"] = []
            collated_dataset.metadata["total_images"] = 0
            return

        versions.sort()
        logger.info(f"Versions to collate: {['v' + str(v) for v in versions]}")

        # Collect all image blobs from untrained versions
        all_image_blobs = []
        for v in versions:
            prefix = f"datasets/Food-11/v{v}/training/"
            version_blobs = list(bucket.list_blobs(prefix=prefix))
            images = [b for b in version_blobs if b.name.lower().endswith((".jpg", ".jpeg", ".png"))]
            logger.info(f"  v{v}: {len(images)} images")
            all_image_blobs.append((v, images))

        total_images = sum(len(imgs) for _, imgs in all_image_blobs)
        logger.info(f"Total images to collate: {total_images}")

        # Flatten and shuffle for splitting
        all_blobs_with_class = []
        for v, images in all_image_blobs:
            for blob in images:
                # blob.name: datasets/Food-11/v1/training/class_03/abc.jpg
                parts = blob.name.split("/")
                class_dir = parts[-2]  # class_03
                filename = parts[-1]   # abc.jpg
                all_blobs_with_class.append((blob, class_dir, f"v{v}_{filename}"))

        random.seed(42)
        random.shuffle(all_blobs_with_class)

        # 80/20 train/val split
        split_idx = int(len(all_blobs_with_class) * 0.8)
        train_set = all_blobs_with_class[:split_idx]
        val_set = all_blobs_with_class[split_idx:]
        logger.info(f"Split: {len(train_set)} training, {len(val_set)} validation")

        # Copy to collated/ directory
        collated_prefix = "collated"

        # Clear any previous collated data
        old_blobs = list(bucket.list_blobs(prefix=f"{collated_prefix}/"))
        if old_blobs:
            logger.info(f"Clearing {len(old_blobs)} old collated files...")
            for b in old_blobs:
                b.delete()

        def copy_set(blob_set, split_name):
            for i, (blob, class_dir, filename) in enumerate(blob_set):
                dest_path = f"{collated_prefix}/{split_name}/{class_dir}/{filename}"
                bucket.copy_blob(blob, bucket, new_name=dest_path)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {split_name}: copied {i + 1}/{len(blob_set)}")

        logger.info("Copying training set...")
        copy_set(train_set, "training")
        logger.info("Copying validation set...")
        copy_set(val_set, "validation")

        # Store metadata for the pipeline
        collated_dataset.metadata["versions_collated"] = versions
        collated_dataset.metadata["total_images"] = total_images
        collated_dataset.metadata["train_count"] = len(train_set)
        collated_dataset.metadata["val_count"] = len(val_set)
        collated_dataset.metadata["gcs_path"] = f"gs://{training_bucket}/{collated_prefix}"
        collated_dataset.uri = f"gs://{training_bucket}/{collated_prefix}"

        logger.info(f"Data preparation complete. Collated at gs://{training_bucket}/{collated_prefix}")

    except Exception as e:
        logger.error(f"prepare_data FAILED: {e}")
        logger.error(traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Component 2: Train Model
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest",
    packages_to_install=["google-cloud-storage", "google-cloud-aiplatform"],
)
def train_model(
    training_bucket: str,
    collated_dataset: Input[Dataset],
    trained_model: Output[Model],
    initial_epochs: int = 5,
    total_epochs: int = 20,
    patience: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    fine_tune_lr: float = 1e-5,
    experiment_name: str = "",
    tensorboard_id: str = "",
    gcp_project: str = "",
    gcp_location: str = "us-central1",
):
    """Train or fine-tune MobileNetV2 on the collated dataset."""
    import json
    import logging
    import os
    import time
    import traceback

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from google.cloud import storage
    from torch.utils.data import DataLoader
    from torchvision import datasets, models, transforms

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    try:
        # Check if there's data to train on
        versions_collated = collated_dataset.metadata.get("versions_collated", [])
        if not versions_collated:
            logger.info("No new data versions to train on. Skipping training.")
            return

        logger.info(f"=== train_model started ===")
        logger.info(f"Training bucket: {training_bucket}")
        logger.info(f"Data versions to train on: {versions_collated}")
        logger.info(f"Hyperparameters: initial_epochs={initial_epochs}, total_epochs={total_epochs}, "
                     f"patience={patience}, batch_size={batch_size}, lr={lr}, fine_tune_lr={fine_tune_lr}")

        client = storage.Client()
        bucket = client.bucket(training_bucket)

        # Download collated data locally
        local_data_dir = "/tmp/collated"
        os.makedirs(local_data_dir, exist_ok=True)

        total_downloaded = 0
        for split in ["training", "validation"]:
            blobs = list(bucket.list_blobs(prefix=f"collated/{split}/"))
            split_count = 0
            for blob in blobs:
                if blob.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    relative = blob.name[len("collated/"):]
                    local_path = os.path.join(local_data_dir, relative)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    blob.download_to_filename(local_path)
                    split_count += 1
                    if split_count % 200 == 0:
                        logger.info(f"  Downloading {split}: {split_count} images...")
            total_downloaded += split_count
            logger.info(f"Downloaded {split} set: {split_count} images")

        logger.info(f"Total images downloaded: {total_downloaded}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if device.type == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(os.path.join(local_data_dir, "training"), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(local_data_dir, "validation"), transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        logger.info(f"Classes ({len(train_dataset.classes)}): {train_dataset.classes}")

        # Build model
        logger.info("Building MobileNetV2 model...")
        model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
        num_ftrs = model.last_channel
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 11),
        )

        # Check for existing trained model to fine-tune
        metadata_blob = bucket.blob("training_metadata.json")
        if metadata_blob.exists():
            metadata = json.loads(metadata_blob.download_as_text())
            last_model_version = metadata.get("last_model_version", 0)
            if last_model_version > 0:
                model_path = f"models/model_v{last_model_version}/food11.pth"
                model_blob = bucket.blob(model_path)
                if model_blob.exists():
                    logger.info(f"Loading existing model: model_v{last_model_version} for fine-tuning")
                    local_model = "/tmp/food11_prev.pth"
                    model_blob.download_to_filename(local_model)
                    state = torch.load(local_model, map_location=device)
                    model.load_state_dict(state)
                    logger.info("Previous model weights loaded successfully.")
                else:
                    logger.info(f"Model file not found at {model_path}. Training from scratch.")
        else:
            metadata = {}
            logger.info("No previous model found. Training from scratch with pretrained MobileNetV2.")

        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {total_params:,}")

        # --- Experiment tracking (optional) ---
        new_model_version = metadata.get("last_model_version", 0) + 1
        tracking = bool(experiment_name)
        if tracking:
            from google.cloud import aiplatform
            logger.info(f"Initializing experiment tracking: {experiment_name}")
            aiplatform.init(
                project=gcp_project,
                location=gcp_location,
                experiment=experiment_name,
                experiment_tensorboard=tensorboard_id if tensorboard_id else None,
            )
            aiplatform.start_run(f"train-v{new_model_version}")
            aiplatform.log_params({
                "initial_epochs": initial_epochs,
                "total_epochs": total_epochs,
                "patience": patience,
                "batch_size": batch_size,
                "learning_rate": lr,
                "fine_tune_lr": fine_tune_lr,
                "data_versions": str(versions_collated),
                "device": str(device),
                "base_model": f"model_v{new_model_version - 1}" if new_model_version > 1 else "pretrained",
            })

        # --- Phase 1: Train classification head only (backbone frozen) ---
        logger.info("=" * 60)
        logger.info("Phase 1: Training classification head (backbone frozen)")
        logger.info("=" * 60)
        for param in model.features.parameters():
            param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

        best_val_loss = float("inf")
        best_model_path = "/tmp/food11_best.pth"

        for epoch in range(initial_epochs):
            start = time.time()
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            t_loss = train_loss / len(train_loader)
            t_acc = train_correct / train_total
            v_loss = val_loss / len(val_loader)
            v_acc = val_correct / val_total
            elapsed = time.time() - start

            logger.info(f"Epoch {epoch+1}/{initial_epochs} | "
                         f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
                         f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | "
                         f"Time: {elapsed:.1f}s")

            if tracking:
                aiplatform.log_time_series_metrics(
                    {"train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc},
                    step=epoch,
                )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                torch.save(model.state_dict(), best_model_path)
                logger.info("  -> Validation loss improved. Model checkpoint saved.")

        # --- Phase 2: Fine-tune entire model (backbone unfrozen) ---
        logger.info("=" * 60)
        logger.info("Phase 2: Fine-tuning entire model (backbone unfrozen)")
        logger.info("=" * 60)
        for param in model.features.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
        patience_counter = 0

        for epoch in range(initial_epochs, total_epochs):
            start = time.time()
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            t_loss = train_loss / len(train_loader)
            t_acc = train_correct / train_total
            v_loss = val_loss / len(val_loader)
            v_acc = val_correct / val_total
            elapsed = time.time() - start

            logger.info(f"Epoch {epoch+1}/{total_epochs} | "
                         f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
                         f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} | "
                         f"Time: {elapsed:.1f}s")

            if tracking:
                aiplatform.log_time_series_metrics(
                    {"train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc},
                    step=epoch,
                )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                logger.info("  -> Validation loss improved. Model checkpoint saved.")
            else:
                patience_counter += 1
                logger.info(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info("  Early stopping triggered.")
                break

        # --- Upload trained model to GCS ---
        logger.info("=" * 60)
        logger.info("Uploading trained model to GCS")
        logger.info("=" * 60)
        last_model_version = new_model_version

        # Versioned model
        versioned_path = f"models/model_v{last_model_version}/food11.pth"
        bucket.blob(versioned_path).upload_from_filename(best_model_path)
        logger.info(f"Versioned model saved: gs://{training_bucket}/{versioned_path}")

        # Latest model
        bucket.blob("models/latest/food11.pth").upload_from_filename(best_model_path)
        logger.info(f"Latest model updated: gs://{training_bucket}/models/latest/food11.pth")

        # Update training metadata
        max_version = max(versions_collated)
        metadata["last_trained_version"] = max_version
        metadata["last_model_version"] = last_model_version
        metadata["trained_on_data_versions"] = versions_collated

        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2), content_type="application/json"
        )
        logger.info(f"Training metadata updated: last_trained_version=v{max_version}, model=model_v{last_model_version}")

        trained_model.uri = f"gs://{training_bucket}/{versioned_path}"
        trained_model.metadata["model_version"] = last_model_version
        trained_model.metadata["data_versions"] = versions_collated

        logger.info(f"Best validation loss: {best_val_loss:.4f}")

        # --- Experiment tracking: log summary + register model ---
        if tracking:
            aiplatform.log_metrics({
                "best_val_loss": best_val_loss,
                "model_version": last_model_version,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            })

            # Register model in Vertex AI Model Registry
            logger.info("Registering model in Vertex AI Model Registry...")
            container_uri = f"{gcp_location}-docker.pkg.dev/{gcp_project}/gourmetgram-repo/gourmetgram"
            aiplatform.Model.upload(
                display_name=f"gourmetgram-model-v{last_model_version}",
                artifact_uri=f"gs://{training_bucket}/models/model_v{last_model_version}",
                serving_container_image_uri=container_uri,
                serving_container_predict_route="/api/predict",
                serving_container_health_route="/test",
                serving_container_ports=[8000],
            )
            logger.info("Model registered in Model Registry.")

            aiplatform.end_run()
            logger.info("Experiment run ended.")

        logger.info("=== train_model completed successfully ===")

    except Exception as e:
        logger.error(f"train_model FAILED: {e}")
        logger.error(traceback.format_exc())
        raise


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
@dsl.pipeline(
    name="gourmetgram-training-pipeline",
    description="Two-stage pipeline: prepare versioned data, then train/fine-tune MobileNetV2",
)
def gourmetgram_training_pipeline(
    training_bucket: str,
    project: str = "",
    location: str = "us-central1",
    initial_epochs: int = 1,
    total_epochs: int = 1,
    patience: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    fine_tune_lr: float = 1e-5,
    experiment_name: str = "",
    tensorboard_id: str = "",
):
    data_task = prepare_data(training_bucket=training_bucket)
    data_task.set_caching_options(False)

    custom_train_job = create_custom_training_job_from_component(
        train_model,
        display_name="gourmetgram-train",
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
    )

    train_task = custom_train_job(
        training_bucket=training_bucket,
        collated_dataset=data_task.outputs["collated_dataset"],
        initial_epochs=initial_epochs,
        total_epochs=total_epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        fine_tune_lr=fine_tune_lr,
        experiment_name=experiment_name,
        tensorboard_id=tensorboard_id,
        gcp_project=project,
        gcp_location=location,
        project=project,
        location=location,
    )


# ---------------------------------------------------------------------------
# CLI: compile and submit
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--training-bucket", required=True)
    parser.add_argument("--pipeline-root", required=True,
                        help="GCS path for pipeline artifacts, e.g. gs://bucket/pipeline-artifacts")
    parser.add_argument("--initial-epochs", type=int, default=1)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--experiment-name", default="",
                        help="Vertex AI experiment name (enables tracking if set)")
    parser.add_argument("--tensorboard-id", default="",
                        help="Vertex AI TensorBoard resource ID (full path)")
    args = parser.parse_args()

    # Step 1: Compile pipeline to YAML template
    from kfp import compiler
    template_path = "pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=gourmetgram_training_pipeline,
        package_path=template_path,
    )
    print(f"Pipeline compiled to {template_path}")

    # Step 2: Upload template to GCS so it's accessible from the UI
    template_gcs_path = f"gs://{args.training_bucket}/pipeline-templates/gourmetgram-training.yaml"
    from google.cloud import storage as gcs
    client = gcs.Client()
    bucket = client.bucket(args.training_bucket)
    blob = bucket.blob("pipeline-templates/gourmetgram-training.yaml")
    blob.upload_from_filename(template_path)
    print(f"Template uploaded to {template_gcs_path}")

    # Step 3: Submit pipeline run
    aiplatform.init(project=args.project, location=args.region)

    job = aiplatform.PipelineJob(
        display_name="gourmetgram-training",
        template_path=template_path,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "training_bucket": args.training_bucket,
            "project": args.project,
            "location": args.region,
            "initial_epochs": args.initial_epochs,
            "total_epochs": args.total_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "fine_tune_lr": args.fine_tune_lr,
            "experiment_name": args.experiment_name,
            "tensorboard_id": args.tensorboard_id,
        },
    )

    job.submit()
    print(f"\nPipeline submitted. View the run at:")
    print(f"  https://console.cloud.google.com/vertex-ai/pipelines/runs?project={args.project}")
    print(f"\nTo re-run this pipeline from the UI with different parameters:")
    print(f"  1. Go to Vertex AI > Pipelines in the console")
    print(f"  2. Click 'Create Run'")
    print(f"  3. Select the template from: {template_gcs_path}")
    print(f"  4. Adjust parameters and submit")
