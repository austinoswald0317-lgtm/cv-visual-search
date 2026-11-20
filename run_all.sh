#!/usr/bin/env bash
# Full pipeline: download model + data, extract embeddings, evaluate.
set -e
cd "$(dirname "$0")"

if [ ! -f models/resnet18-v1-7.onnx ]; then
  echo "Downloading ResNet-18 (ONNX Model Zoo)..."
  mkdir -p models
  curl -L -o models/resnet18-v1-7.onnx \
    https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx
fi

if [ ! -f data/cifar10-test.parquet ]; then
  echo "Downloading CIFAR-10 test split (Hugging Face mirror)..."
  mkdir -p data
  curl -L -o data/cifar10-test.parquet \
    "https://huggingface.co/api/datasets/uoft-cs/cifar10/parquet/plain_text/test/0.parquet"
fi

.venv/bin/python src/extract_embeddings.py
.venv/bin/python src/evaluate.py
echo "Done. See results/results.json and results/demo.html"
