"""Search the gallery with any image file.

Usage:
    .venv/bin/python src/search.py --image path/to/query.jpg --top 5
"""
import argparse
import os

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    data = np.load(os.path.join(ROOT, "results", "embeddings.npz"), allow_pickle=True)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(
        os.path.join(ROOT, "models", "resnet18-embed.onnx"),
        sess_options=opts, providers=["CPUExecutionProvider"])
    img = Image.open(args.image).convert("RGB").resize((224, 224), Image.BICUBIC)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, :]
    emb = session.run(None, {session.get_inputs()[0].name: arr})[0].reshape(1, -1)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    sims = (data["gallery_emb"] @ emb.T).ravel()
    top = np.argsort(-sims)[:args.top]
    names = [str(x) for x in data["class_names"]]
    print(f"Query: {args.image}")
    for rank, j in enumerate(top, 1):
        print(f"  {rank}. gallery #{j} — {names[int(data['gallery_labels'][j])]}"
              f" (cosine {sims[j]:.3f})")


if __name__ == "__main__":
    main()
