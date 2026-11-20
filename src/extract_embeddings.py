"""Extract ResNet-18 embeddings (and a colour-histogram baseline) for CIFAR-10.

Gallery: first 5,000 images of the CIFAR-10 test split.
Queries:  next 500 images (disjoint from the gallery).

The embedding is the 512-d output of ResNet-18's global average pooling layer
(the classification head is removed), L2-normalised. The model is the ONNX
Model Zoo's ResNet-18 v1-7, pretrained on ImageNet; we build a second ONNX
graph whose only output is the pooling layer, so no retraining is needed.

Outputs: results/embeddings.npz
"""
import io
import os

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
from PIL import Image
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "resnet18-v1-7.onnx")
EMBED_MODEL = os.path.join(ROOT, "models", "resnet18-embed.onnx")
DATA_PARQUET = os.path.join(ROOT, "data", "cifar10-test.parquet")
OUT = os.path.join(ROOT, "results", "embeddings.npz")

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
N_GALLERY = 5000
N_QUERY = 500
BATCH = 64

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _quiet_options():
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # hide harmless "initializer in graph inputs" warnings
    return opts


def build_embedding_model() -> str:
    """Create an ONNX model that outputs the global-average-pooling layer."""
    if os.path.exists(EMBED_MODEL):
        return EMBED_MODEL
    model = onnx.load(MODEL)
    gap = next(n for n in model.graph.node if n.op_type == "GlobalAveragePool")
    gap_out = gap.output[0]
    # Keep all inputs, replace outputs with the pooling output.
    new_graph = helper.make_graph(
        model.graph.node, model.graph.name,
        model.graph.input,
        [helper.make_tensor_value_info(gap_out, onnx.TensorProto.FLOAT, [None, 512, 1, 1])],
        initializer=model.graph.initializer,
    )
    onnx.save(helper.make_model(new_graph, opset_imports=model.opset_import),
              EMBED_MODEL)
    print(f"Embedding model written to {EMBED_MODEL} (output: {gap_out})")
    return EMBED_MODEL


def load_cifar10_test():
    """CIFAR-10 test split from the Hugging Face parquet mirror
    (uoft-cs/cifar10, plain_text/test) — same images/labels as the
    Toronto tarball, but downloadable quickly. Returns uint8
    [N,32,32,3] images and int labels."""
    table = pq.read_table(DATA_PARQUET)
    labels = np.array(table.column("label").to_pylist())
    blobs = table.column("img").to_pylist()
    images = np.empty((len(blobs), 32, 32, 3), dtype=np.uint8)
    for i, row in enumerate(blobs):
        images[i] = np.asarray(Image.open(io.BytesIO(row["bytes"])).convert("RGB"))
    return images, labels


def preprocess(images: np.ndarray) -> np.ndarray:
    """uint8 [N,32,32,3] -> float32 [N,3,224,224], ImageNet-normalised."""
    out = np.empty((len(images), 3, 224, 224), dtype=np.float32)
    for i, img in enumerate(images):
        pil = Image.fromarray(img).resize((224, 224), Image.BICUBIC)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        out[i] = ((arr - MEAN) / STD).transpose(2, 0, 1)
    return out


def colour_histogram(images: np.ndarray) -> np.ndarray:
    """64-d RGB histogram (4 bins/channel), L2-normalised. Baseline feature."""
    feats = np.empty((len(images), 64), dtype=np.float32)
    for i, img in enumerate(images):
        h, _ = np.histogramdd(
            img.reshape(-1, 3), bins=4, range=[(0, 256)] * 3)
        feats[i] = h.ravel()
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return feats


def embed(session: ort.InferenceSession, images: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    embs = []
    for s in range(0, len(images), BATCH):
        batch = preprocess(images[s:s + BATCH])
        out = session.run(None, {name: batch})[0]  # [B, 512, 1, 1]
        e = out.reshape(len(batch), -1).astype(np.float32)
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        embs.append(e)
        print(f"  embedded {min(s + BATCH, len(images))}/{len(images)}")
    return np.concatenate(embs)


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    embed_path = build_embedding_model()
    session = ort.InferenceSession(
        embed_path,
        sess_options=_quiet_options(),
        providers=["CPUExecutionProvider"])
    images, labels = load_cifar10_test()
    gallery_img, gallery_lab = images[:N_GALLERY], labels[:N_GALLERY]
    query_img, query_lab = images[N_GALLERY:N_GALLERY + N_QUERY], labels[N_GALLERY:N_GALLERY + N_QUERY]

    print("Extracting gallery embeddings...")
    gallery_emb = embed(session, gallery_img)
    print("Extracting query embeddings...")
    query_emb = embed(session, query_img)

    np.savez_compressed(
        OUT,
        gallery_emb=gallery_emb, query_emb=query_emb,
        gallery_hist=colour_histogram(gallery_img),
        query_hist=colour_histogram(query_img),
        gallery_labels=gallery_lab, query_labels=query_lab,
        gallery_images=gallery_img, query_images=query_img,
        class_names=np.array(CLASS_NAMES),
    )
    print(f"Saved {OUT}")
    print(f"gallery {gallery_emb.shape}, queries {query_emb.shape}")


if __name__ == "__main__":
    main()
