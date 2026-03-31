"""Minimal test pipeline to debug Vertex AI Pipeline execution."""
import argparse

from google.cloud import aiplatform
from kfp import compiler, dsl


@dsl.component(base_image="python:3.11-slim")
def hello_world(message: str) -> str:
    print(f"Hello from Vertex AI Pipeline: {message}")
    return message


@dsl.pipeline(name="test-pipeline")
def test_pipeline(message: str = "it works!"):
    hello_world(message=message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--pipeline-root", required=True)
    args = parser.parse_args()

    compiler.Compiler().compile(
        pipeline_func=test_pipeline,
        package_path="test_pipeline.yaml",
    )

    aiplatform.init(project=args.project, location=args.region)

    job = aiplatform.PipelineJob(
        display_name="test-pipeline",
        template_path="test_pipeline.yaml",
        pipeline_root=args.pipeline_root,
    )
    job.submit()
    print("Test pipeline submitted.")
