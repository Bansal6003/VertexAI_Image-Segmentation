"""
Submit a Vertex AI CustomContainerTrainingJob
=============================================
Run this script AFTER building and pushing Docker images with build_and_push.ps1.

Usage:
  python submit_training_job.py

The script blocks until the job completes, then prints the GCS URI of the saved model.
"""

from google.cloud import aiplatform

# ============================================================
# FILL IN THESE VALUES
# ============================================================
PROJECT_ID     = "project-name"
REGION         = "us-central1"
BUCKET         = "larva-segmennt_training"          # without gs://
IMAGES_PREFIX  = "larva_images"               # GCS prefix inside bucket
MASKS_PREFIX   = "larva_masks"                # GCS prefix inside bucket

# URI from build_and_push.ps1 output
TRAIN_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/larva-seg/larva-seg-train:latest"
)

# Where Vertex AI will save model artifacts
MODEL_GCS_DIR = f"gs://{BUCKET}/vertex-training/model"

MACHINE_TYPE      = "a2-ultragpu-2g"
ACCELERATOR       = "NVIDIA_A100_80GB"
ACCELERATOR_COUNT = 2

# ============================================================

# Training hyperparameters — passed as CLI args to train.py
ARGS = [
    "--bucket",          BUCKET,
    "--images-prefix",   IMAGES_PREFIX,
    "--masks-prefix",    MASKS_PREFIX,
    "--encoder",         "resnet18",
    "--encoder-weights", "imagenet",
    "--epochs",          "300",
    "--batch-size",      "64",
    "--lr",              "1e-4",
    "--weight-decay",    "1e-6",
    "--img-size",        "512",
    "--val-split",       "0.2",
    "--patience",        "10",
    "--num-workers",     "4",
]


def main():
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET}/staging")

    job = aiplatform.CustomContainerTrainingJob(
        display_name="larva-seg-custom-container",
        container_uri=TRAIN_IMAGE_URI,
        model_serving_container_image_uri=None,  # we handle serving separately
    )

    run_kwargs = dict(
        args=ARGS,
        base_output_dir=MODEL_GCS_DIR,
        machine_type=MACHINE_TYPE,
        replica_count=1,
        sync=True,
    )
    if ACCELERATOR_COUNT > 0:
        run_kwargs["accelerator_type"] = ACCELERATOR
        run_kwargs["accelerator_count"] = ACCELERATOR_COUNT

    model = job.run(**run_kwargs)

    print("\n=== Training complete ===")
    print(f"Model artifacts: {MODEL_GCS_DIR}")
    print(f"Model resource : {model.resource_name if model else 'N/A'}")
    print("\nNext step: run deploy_model.py")


if __name__ == "__main__":
    main()
