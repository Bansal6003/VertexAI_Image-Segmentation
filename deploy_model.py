"""
Deploy trained model to a Vertex AI Endpoint
=============================================
Run this AFTER submit_training_job.py finishes.

The serving container (Dockerfile.serve) is used for online prediction.
It reads MODEL_PATH from its environment variable, which we set to the
GCS URI of model.onnx produced by training.

WARNING: A deployed Vertex AI endpoint costs ~$0.19/hr minimum even with
zero traffic. Delete the endpoint when you are done testing.
"""

from google.cloud import aiplatform

# ============================================================
# FILL IN THESE VALUES
# ============================================================
PROJECT_ID   = "project-4484f804-b862-439c-835"
REGION       = "us-central1"
BUCKET       = "larva-segmennt_training"

# ONNX model artifact produced by training (inside MODEL_GCS_DIR/model.onnx)
MODEL_ONNX_GCS = f"gs://{BUCKET}/vertex-training/model/model.onnx"

# Serving image URI from build_and_push.ps1
SERVE_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/larva-seg/larva-seg-serve:latest"
)

MACHINE_TYPE     = "n1-standard-2"   # CPU serving is fine for ONNX
ENDPOINT_NAME    = "larva-seg-endpoint"
MODEL_DISPLAY    = "larva-seg-deeplabv3plus-onnx"
MIN_REPLICA      = 1
MAX_REPLICA      = 2
# ============================================================


def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    # --- Upload model to Vertex AI Model Registry ---
    print("Uploading model to Vertex AI Model Registry...")
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY,
        artifact_uri=f"gs://{BUCKET}/vertex-training/model/",  # directory containing model.onnx
        serving_container_image_uri=SERVE_IMAGE_URI,
        serving_container_environment_variables={
            "MODEL_PATH": MODEL_ONNX_GCS,
        },
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )
    print(f"Model uploaded: {model.resource_name}")

    # --- Create or get endpoint ---
    print(f"\nCreating endpoint '{ENDPOINT_NAME}'...")
    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_NAME)
    print(f"Endpoint: {endpoint.resource_name}")

    # --- Deploy model to endpoint ---
    print("\nDeploying model to endpoint (this takes 5-10 minutes)...")
    endpoint.deploy(
        model=model,
        machine_type=MACHINE_TYPE,
        min_replica_count=MIN_REPLICA,
        max_replica_count=MAX_REPLICA,
        traffic_percentage=100,
        sync=True,
    )

    print("\n=== Deployment complete ===")
    print(f"Endpoint resource: {endpoint.resource_name}")
    print(f"\nTo test: python run_inference.py")
    print(f"\n!! Remember to delete the endpoint when done to avoid charges !!")
    print(f"   endpoint.delete(force=True)  or use deploy_model.py --delete")


def delete_endpoint(endpoint_resource_name: str):
    """Helper to delete endpoint and undeploy all models."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(endpoint_resource_name)
    endpoint.delete(force=True)
    print(f"Endpoint {endpoint_resource_name} deleted.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--delete":
        if len(sys.argv) < 3:
            print("Usage: python deploy_model.py --delete <endpoint_resource_name>")
            sys.exit(1)
        delete_endpoint(sys.argv[2])
    else:
        main()
