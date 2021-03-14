#!/bin/bash
cat <<EOT > .env
UID=`id -u`
GID=`id -g`
MINIO_ACCESS_KEY="minio-access-key"
MINIO_SECRET_KEY="minio-secret-key"
MLFLOW_S3_ENDPOINT_URL="http://minio:9000"
AWS_ACCESS_KEY_ID="minio-access-key"
AWS_SECRET_ACCESS_KEY="minio-access-key"
EOT
