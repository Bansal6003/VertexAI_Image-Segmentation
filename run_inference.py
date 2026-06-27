"""
Run inference against a deployed Vertex AI Endpoint
====================================================
Two modes:
  1. Online prediction via Vertex AI endpoint (requires deployed endpoint)
  2. Local ONNX inference (for quick testing without deployment)

Usage:
  # Vertex AI endpoint
  python run_inference.py --image path/to/image.jpg --endpoint ENDPOINT_RESOURCE_NAME

  # Local ONNX
  python run_inference.py --image path/to/image.jpg --onnx path/to/model.onnx --local
"""

import argparse
import base64
import io
import json
from pathlib import Path

from PIL import Image


# ============================================================
# FILL IN THESE VALUES
# ============================================================
PROJECT_ID = "project-4484f804-b862-439c-835"
REGION     = "us-east4"
# ============================================================


def predict_endpoint(image_path: str, endpoint_resource_name: str):
    from google.cloud import aiplatform

    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint(endpoint_resource_name)

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    response = endpoint.predict(instances=[{"image_bytes": b64}])
    return response.predictions[0]


def predict_local(image_path: str, onnx_path: str):
    import numpy as np
    import onnxruntime as ort

    SIZE = 512
    img = Image.open(image_path).convert("L").resize((SIZE, SIZE))
    arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    inp = np.stack([arr, arr, arr], axis=0)[np.newaxis].astype(np.float32)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    raw = session.run([output_name], {input_name: inp})[0]

    prob = 1.0 / (1.0 + np.exp(-raw))
    mask = (prob > 0.5).astype(np.uint8)[0, 0] * 255

    n_fg = int((mask > 0).sum())
    total = mask.size
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")

    return {
        "mask_base64": base64.b64encode(buf.getvalue()).decode(),
        "foreground_pixels": n_fg,
        "total_pixels": total,
        "coverage_ratio": round(n_fg / total, 4),
    }


def save_result(result: dict, out_dir: str = "inference_output"):
    Path(out_dir).mkdir(exist_ok=True)

    mask_bytes = base64.b64decode(result["mask_base64"])
    mask_img = Image.open(io.BytesIO(mask_bytes))
    mask_path = Path(out_dir) / "predicted_mask.png"
    mask_img.save(mask_path)

    meta_path = Path(out_dir) / "result.json"
    meta = {k: v for k, v in result.items() if k != "mask_base64"}
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Mask saved   : {mask_path}")
    print(f"Metadata     : {meta_path}")
    print(f"Coverage     : {result['coverage_ratio']*100:.1f}%")
    print(f"Foreground px: {result['foreground_pixels']:,} / {result['total_pixels']:,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--endpoint", help="Vertex AI endpoint resource name (projects/.../endpoints/...)")
    p.add_argument("--onnx", help="Local ONNX model path (for --local mode)")
    p.add_argument("--local", action="store_true", help="Run locally with ONNX instead of endpoint")
    p.add_argument("--out-dir", default="inference_output")
    args = p.parse_args()

    if args.local:
        if not args.onnx:
            raise ValueError("--onnx is required with --local")
        print(f"Running local ONNX inference on {args.image}...")
        result = predict_local(args.image, args.onnx)
    else:
        if not args.endpoint:
            raise ValueError("--endpoint is required for online prediction")
        print(f"Running Vertex AI endpoint prediction...")
        result = predict_endpoint(args.image, args.endpoint)

    save_result(result, args.out_dir)


if __name__ == "__main__":
    main()
