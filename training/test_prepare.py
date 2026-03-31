"""Test pipeline with just the prepare_data component to isolate the error."""
import argparse

from google.cloud import aiplatform
from kfp import compiler, dsl
from kfp.dsl import Dataset, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-storage"],
)
def prepare_data_test(
    training_bucket: str,
    collated_dataset: Output[Dataset],
):
    """Minimal version of prepare_data to find the error."""
    import json
    import logging
    from google.cloud import storage

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Starting prepare_data_test with bucket: {training_bucket}")

    client = storage.Client()
    bucket = client.bucket(training_bucket)

    logger.info("Listing versions...")
    blobs = bucket.list_blobs(prefix="datasets/Food-11/v", delimiter="/")
    list(blobs)
    versions = []
    for prefix in blobs.prefixes:
        folder_name = prefix.rstrip("/").split("/")[-1]
        if folder_name.startswith("v") and folder_name[1:].isdigit():
            versions.append(int(folder_name[1:]))

    logger.info(f"Found versions: {versions}")

    collated_dataset.metadata["versions_found"] = versions
    collated_dataset.uri = f"gs://{training_bucket}/collated"
    logger.info("Done!")


@dsl.pipeline(name="test-prepare-data")
def test_pipeline(training_bucket: str):
    prepare_data_test(training_bucket=training_bucket)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--training-bucket", required=True)
    parser.add_argument("--pipeline-root", required=True)
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=test_pipeline,
        package_path="test_prepare.yaml",
    )

    aiplatform.init(project=args.project, location=args.region)

    job = aiplatform.PipelineJob(
        display_name="test-prepare-data",
        template_path="test_prepare.yaml",
        pipeline_root=args.pipeline_root,
        parameter_values={"training_bucket": args.training_bucket},
    )
    job.submit()
    print("Test prepare pipeline submitted.")
