# ============================================================
# Build and push both Docker images to Artifact Registry
# Run from the vertex_ai_custom_container/ directory
#
# FILL IN THESE VALUES BEFORE RUNNING:
# ============================================================
$PROJECT_ID   = "project-name"
$REGION       = "us-central1"
$REPO_NAME    = "larva-seg"         # Artifact Registry repo name (will be created)
$IMAGE_TRAIN  = "larva-seg-train"
$IMAGE_SERVE  = "larva-seg-serve"
$TAG          = "latest"

$REGISTRY = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"

# ---- 1. Authenticate Docker with Artifact Registry ----
Write-Host "Configuring Docker auth for $REGION..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# ---- 2. Create Artifact Registry repo (idempotent) ----
Write-Host "Creating Artifact Registry repo '$REPO_NAME' (if not exists)..."
gcloud artifacts repositories create $REPO_NAME `
    --repository-format=docker `
    --location=$REGION `
    --description="Larva segmentation images" `
    --project=$PROJECT_ID 2>$null
# ignore error if repo already exists

# ---- 3. Build & push training image ----
Write-Host "`nBuilding training image..."
docker build -f Dockerfile.train -t "$REGISTRY/${IMAGE_TRAIN}:$TAG" .
if ($LASTEXITCODE -ne 0) { Write-Error "Training image build failed"; exit 1 }

Write-Host "Pushing training image..."
docker push "$REGISTRY/${IMAGE_TRAIN}:$TAG"
if ($LASTEXITCODE -ne 0) { Write-Error "Training image push failed"; exit 1 }

# ---- 4. Build & push serving image ----
Write-Host "`nBuilding serving image..."
docker build -f Dockerfile.serve -t "$REGISTRY/${IMAGE_SERVE}:$TAG" .
if ($LASTEXITCODE -ne 0) { Write-Error "Serving image build failed"; exit 1 }

Write-Host "Pushing serving image..."
docker push "$REGISTRY/${IMAGE_SERVE}:$TAG"
if ($LASTEXITCODE -ne 0) { Write-Error "Serving image push failed"; exit 1 }

Write-Host "`n=== Done ==="
Write-Host "Training image : $REGISTRY/${IMAGE_TRAIN}:$TAG"
Write-Host "Serving image  : $REGISTRY/${IMAGE_SERVE}:$TAG"
Write-Host "`nCopy these URIs into submit_training_job.py and deploy_model.py"
