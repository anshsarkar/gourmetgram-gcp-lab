#!/bin/bash
# Cleanup script for GourmetGram GCP Lab
# Run from Cloud Shell: bash cleanup.sh

set -e

# --- Configuration ---
export GCP_PROJECT_ID="gourmetgram-gcp-lab"
export REGION="us-central1"
export ZONE="us-central1-a"

# IMPORTANT: Change this to your Net ID
export NET_ID="as20363"

export GCS_STAGING_BUCKET="${NET_ID}-staging-bucket"
export GCS_TRAINING_BUCKET="${NET_ID}-training-bucket"
export GCS_EVENTARC_BUCKET="${NET_ID}-eventarc-bucket"
export SERVICE_NAME="gourmetgram-service"
export CLUSTER_NAME="gourmetgram-cluster"

echo "=== GourmetGram GCP Lab Cleanup ==="
echo "Project: $GCP_PROJECT_ID"
echo "Net ID:  $NET_ID"
echo ""

# --- Cloud Scheduler ---
echo "[1/10] Deleting Cloud Scheduler job..."
gcloud scheduler jobs delete batch-data-scheduler --location=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- Cloud Run Job ---
echo "[2/10] Deleting Cloud Run Job..."
gcloud run jobs delete batch-data-job --region=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- Eventarc Trigger ---
echo "[3/10] Deleting Eventarc trigger..."
gcloud eventarc triggers delete gourmetgram-eventarc-trigger --location=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- Cloud Run Service ---
echo "[4/10] Deleting Cloud Run service..."
gcloud run services delete $SERVICE_NAME --region=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- VM Instance ---
echo "[5/10] Deleting VM instance..."
gcloud compute instances delete gourmetgram-vm --zone=$ZONE --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- Firewall Rule ---
echo "[6/10] Deleting firewall rule..."
gcloud compute firewall-rules delete allow-gourmetgram-vm --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- GKE Cluster (takes a few minutes) ---
echo "[7/10] Deleting GKE cluster (this may take a few minutes)..."
gcloud container clusters delete $CLUSTER_NAME --region=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- GCS Buckets ---
echo "[8/10] Deleting GCS buckets..."
gsutil rm -r gs://$GCS_STAGING_BUCKET 2>/dev/null || echo "  Staging bucket not found, skipping"
gsutil rm -r gs://$GCS_TRAINING_BUCKET 2>/dev/null || echo "  Training bucket not found, skipping"
gsutil rm -r gs://$GCS_EVENTARC_BUCKET 2>/dev/null || echo "  Eventarc bucket not found, skipping"

# --- Vertex AI Endpoint ---
echo "[9/14] Deleting Vertex AI Endpoint (undeploys all models first)..."
ENDPOINT_ID=$(gcloud ai endpoints list --region=$REGION --format="value(name)" --filter="displayName=gourmetgram-endpoint" --limit=1 2>/dev/null)
if [ -n "$ENDPOINT_ID" ]; then
  # Undeploy each model individually (--all flag doesn't exist)
  for DM_ID in $(gcloud ai endpoints describe $ENDPOINT_ID --region=$REGION --format="json" 2>/dev/null | python3 -c "import sys,json; [print(m['id']) for m in json.loads(sys.stdin.read()).get('deployedModels',[])]" 2>/dev/null); do
    gcloud ai endpoints undeploy-model $ENDPOINT_ID --deployed-model-id=$DM_ID --region=$REGION --quiet 2>/dev/null || true
  done
  gcloud ai endpoints delete $ENDPOINT_ID --region=$REGION --quiet 2>/dev/null || true
else
  echo "  (not found, skipping)"
fi

# --- Vertex AI Models ---
echo "[10/14] Deleting Vertex AI models..."
for MODEL in $(gcloud ai models list --region=$REGION --format="value(name)" --filter="displayName~gourmetgram" 2>/dev/null); do
  gcloud ai models delete $MODEL --region=$REGION --quiet 2>/dev/null || true
done

# --- Vertex AI TensorBoard ---
echo "[11/14] Deleting Vertex AI TensorBoard instances..."
for TB in $(gcloud ai tensorboards list --region=$REGION --format="value(name)" 2>/dev/null); do
  gcloud ai tensorboards delete $TB --region=$REGION --quiet 2>/dev/null || true
done

# --- Vertex AI Experiments ---
echo "[12/14] Deleting Vertex AI Experiments..."
gcloud ai experiments delete gourmetgram-experiment --region=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- Artifact Registry Images ---
echo "[13/17] Deleting Artifact Registry images..."
gcloud artifacts docker images delete $REGION-docker.pkg.dev/$GCP_PROJECT_ID/gourmetgram-repo/gourmetgram --delete-tags --quiet 2>/dev/null || echo "  gourmetgram image not found, skipping"
gcloud artifacts docker images delete $REGION-docker.pkg.dev/$GCP_PROJECT_ID/gourmetgram-repo/batch-data-job --delete-tags --quiet 2>/dev/null || echo "  batch-data-job image not found, skipping"
gcloud artifacts docker images delete $REGION-docker.pkg.dev/$GCP_PROJECT_ID/gourmetgram-repo/gourmetgram-training --delete-tags --quiet 2>/dev/null || echo "  gourmetgram-training image not found, skipping"

# --- Artifact Registry Repo ---
echo "[14/17] Deleting Artifact Registry repository..."
gcloud artifacts repositories delete gourmetgram-repo --location=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

# --- Monitoring Dashboard ---
echo "[15/17] Deleting monitoring dashboard..."
DASHBOARD_ID=$(gcloud monitoring dashboards list --format="value(name)" --filter="displayName='GourmetGram Overview'" 2>/dev/null | head -1)
if [ -n "$DASHBOARD_ID" ]; then
  gcloud monitoring dashboards delete "$DASHBOARD_ID" --quiet 2>/dev/null || true
else
  echo "  (not found, skipping)"
fi

# --- Alerting Policy ---
echo "[16/17] Deleting alerting policy..."
POLICY_ID=$(gcloud alpha monitoring policies list --format="value(name)" --filter="displayName='GourmetGram High Latency Alert'" 2>/dev/null | head -1)
if [ -n "$POLICY_ID" ]; then
  gcloud alpha monitoring policies delete "$POLICY_ID" --quiet 2>/dev/null || true
else
  echo "  (not found, skipping)"
fi

# --- Log-based Metric ---
echo "[17/17] Deleting log-based metric..."
gcloud logging metrics delete prediction_request_count --quiet 2>/dev/null || echo "  (not found, skipping)"

echo ""
echo "=== Cleanup complete ==="
echo "Check the console to verify: https://console.cloud.google.com/home/dashboard?project=$GCP_PROJECT_ID"