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

# --- Artifact Registry Images ---
echo "[9/10] Deleting Artifact Registry images..."
gcloud artifacts docker images delete $REGION-docker.pkg.dev/$GCP_PROJECT_ID/gourmetgram-repo/gourmetgram --delete-tags --quiet 2>/dev/null || echo "  gourmetgram image not found, skipping"
gcloud artifacts docker images delete $REGION-docker.pkg.dev/$GCP_PROJECT_ID/gourmetgram-repo/batch-data-job --delete-tags --quiet 2>/dev/null || echo "  batch-data-job image not found, skipping"
gcloud artifacts docker images delete $REGION-docker.pkg.dev/$GCP_PROJECT_ID/gourmetgram-repo/gourmetgram-training --delete-tags --quiet 2>/dev/null || echo "  gourmetgram-training image not found, skipping"

# --- Artifact Registry Repo ---
echo "[10/10] Deleting Artifact Registry repository..."
gcloud artifacts repositories delete gourmetgram-repo --location=$REGION --quiet 2>/dev/null || echo "  (not found, skipping)"

echo ""
echo "=== Cleanup complete ==="
echo "Check the console to verify: https://console.cloud.google.com/home/dashboard?project=$GCP_PROJECT_ID"