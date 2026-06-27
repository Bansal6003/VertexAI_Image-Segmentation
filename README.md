# Vertex AI — Custom Container Training

Full custom Docker container approach. You own the entire environment.

## Directory structure

```
vertex_ai_custom_container/
├── trainer/
│   ├── train.py        ← training entrypoint (ENTRYPOINT in Dockerfile)
│   ├── dataset.py      ← GCS download + Dataset
│   ├── model.py        ← DeepLabV3Plus factory
│   ├── losses.py       ← ComboLoss (BCE + Dice + Focal)
│   └── metrics.py      ← Dice, IoU, F1, etc.
├── serving/
│   ├── app.py          ← FastAPI prediction server
│   ├── predictor.py    ← ONNX inference
│   └── requirements.txt
├── Dockerfile.train    ← CUDA training image
├── Dockerfile.serve    ← lightweight ONNX serving image
├── requirements.train.txt
├── build_and_push.ps1  ← Step 1
├── submit_training_job.py  ← Step 2
├── deploy_model.py     ← Step 3
└── run_inference.py    ← Step 4
```

## Step-by-step execution order

### Prerequisites
```powershell
pip install google-cloud-aiplatform google-cloud-storage
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud config set project YOUR_PROJECT_ID
```

> **If you get a 403 Forbidden or "no valid credentials" error** when pushing Docker images, re-run:
> ```powershell
> gcloud auth login
> gcloud auth configure-docker us-central1-docker.pkg.dev
> ```
> Then re-run the build script. This re-authenticates Docker with Artifact Registry.

### 0. Fill in your values
Edit the top of every `.ps1` and `.py` file and replace:
- `YOUR_PROJECT_ID` — your GCP project ID (`project-4484f804-b862-439c-835`)
- `YOUR_BUCKET_NAME` — your GCS bucket name (no `gs://`)
- `larva_images` — GCS folder containing your images
- `larva_masks`  — GCS folder containing your masks

### 1. Build & push Docker images
```powershell
cd vertex_ai_custom_container
.\build_and_push.ps1
```
Pushes two images to Artifact Registry:
- `larva-seg-train:latest` — CUDA training container
- `larva-seg-serve:latest` — ONNX serving container

### 2. Submit training job
```powershell
python submit_training_job.py
```
- Runs on Vertex AI with T4 GPU
- Downloads your images/masks from GCS at startup
- Saves `best_model.pth`, `final_model.pth`, `model.onnx` to GCS

### 3. Deploy to Vertex AI endpoint
```powershell
python deploy_model.py
```
- Registers the ONNX model in Vertex AI Model Registry
- Deploys to an online prediction endpoint
- **WARNING**: costs ~$0.19/hr — delete when done

### 4. Run inference
```powershell
# Against deployed endpoint
python run_inference.py --image path/to/larva.jpg --endpoint projects/.../endpoints/...

# Local ONNX (no deployment needed — for testing)
python run_inference.py --local --onnx /tmp/model.onnx --image path/to/larva.jpg
```

### Delete endpoint when done
```powershell
python deploy_model.py --delete projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID
```

## Key training args
| Arg | Default | Notes |
|-----|---------|-------|
| `--bucket` | required | GCS bucket name |
| `--images-prefix` | required | GCS prefix for images |
| `--masks-prefix` | required | GCS prefix for masks |
| `--encoder` | `resnet18` | SMP encoder name |
| `--epochs` | 100 | Max epochs |
| `--batch-size` | 16 | Reduce to 8 if OOM |
| `--lr` | 1e-4 | Learning rate |
| `--patience` | 50 | Early stopping |

## GCS data structure expected
```
gs://YOUR_BUCKET/
├── data/images/
│   ├── larva_001.jpg
│   └── ...
└── data/masks/
    ├── larva_001.png       (stem matches image, OR)
    ├── larva_001_mask_0.png  (_mask_0 suffix auto-stripped)
    └── ...
```

## Metrics logged
All metrics are printed to stdout (captured by Vertex AI Logs) and optionally
logged to Vertex AI Experiments if `google-cloud-aiplatform` is available:
- `train/loss`, `train/dice`, `train/iou`
- `val/loss`, `val/dice`, `val/iou`, `val/f1`, `val/accuracy`
- `val/sensitivity`, `val/specificity`, `val/precision`
- `learning_rate`
