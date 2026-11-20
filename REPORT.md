# Visual Search with Pretrained ResNet-18 Embeddings — Short Report

**Author:** Austin, BSc Computer Science + Data Science, University of Sydney
**Date:** November 2026
**Code:** see `README.md` in this folder (reproducible: `bash run_all.sh`)

## 1. Problem

Given a query image, retrieve the most visually similar images from a gallery.
This is the core task of *visual search*, one of the research directions of
Dr Chang Xu's group at the University of Sydney.

## 2. Method

- **Representation:** ResNet-18 pretrained on ImageNet (ONNX Model Zoo,
  `resnet18-v1-7`). The classification head is removed and the 512-dimensional
  output of global average pooling is used as the image embedding, L2-normalised.
  No training from scratch — this is transfer learning: features learned on
  ImageNet are reused on a new dataset.
- **Search:** exact nearest neighbours by cosine similarity over the gallery
  embeddings (brute force with NumPy; exact, no approximation).
- **Data:** CIFAR-10 test split — gallery 5,000 images, queries 500 images,
  disjoint. Ten classes (airplane, automobile, bird, cat, deer, dog, frog,
  horse, ship, truck).
- **Baseline:** a 64-d RGB colour histogram, searched identically, to measure
  what the deep representation adds over a hand-crafted feature.

## 3. Results

| Method | Recall@5 | Recall@10 | mAP@10 |
|---|---|---|---|
| ResNet-18 embeddings | 0.940 | 0.962 | 0.770 |
| Colour-histogram baseline | 0.622 | 0.804 | 0.337 |

Per-class Recall@10 for ResNet-18: airplane 1.00, automobile 1.00, truck 1.00,
horse 0.98, ship 0.97, frog 0.94, deer 0.94, dog 0.94, cat 0.92, bird 0.90 —
hardest classes are the visually similar animal pairs (cat/dog, bird).

Relevance = retrieved image has the same CIFAR-10 class as the query.
Example queries with top-5 results: `results/demo.html`.

## 4. Discussion

- The deep embedding beats the colour baseline by a wide margin, because it
  encodes shape and object structure, not just colour statistics — e.g. a red
  car and a red bird share colours but not an embedding neighbourhood.
- Errors are concentrated where classes are visually close (cat vs. dog,
  automobile vs. truck); see per-class Recall@10 in `results/results.json`.

## 5. Limitations (stated honestly)

- Same-class is only a proxy for visual similarity; two images of the same
  class can look very different, and two of different classes can look alike.
  Numbers and qualitative examples should be read together.
- CIFAR-10 images are tiny (32x32, upscaled to 224x224 for the network), so
  fine detail is lost.
- Brute-force search does not scale; a production system would use an
  approximate index (e.g. FAISS).

## 6. What I would do next

Fine-tune the backbone on CIFAR-10, compare a stronger backbone (ResNet-50 /
ViT), and replace brute-force search with an approximate nearest-neighbour
index so the gallery can grow to millions of images.
