

ECR_URL=439440559013.dkr.ecr.ap-southeast-2.amazonaws.com
REPO_URL=${ECR_URL}/clothing-prediction
REMOTE_IMAGE_TAG="${REPO_URL}:v1"

LOCAL_IMAGE=clothing-pytorch


aws ecr get-login-password \
  --region "ap-southeast-2" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}


docker build --platform linux/amd64 --provenance false -t ${LOCAL_IMAGE} .
docker tag ${LOCAL_IMAGE} ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "Done"