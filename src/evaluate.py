"""Evaluate visual search: cosine retrieval, Recall@k / mAP@k, baseline comparison.

Relevance proxy: a retrieved image counts as relevant when its CIFAR-10 class
matches the query's class. This is a proxy, not ground-truth visual similarity
— the report states this limitation explicitly.

Outputs: results/results.json, results/demo.html, results/examples.png
"""
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, "results", "embeddings.npz")


def recall_at_k(ranked_labels, true_label, k):
    return float(np.any(ranked_labels[:k] == true_label))


def ap_at_k(ranked_labels, true_label, k):
    rel = (ranked_labels[:k] == true_label).astype(float)
    if rel.sum() == 0:
        return 0.0
    precisions = np.cumsum(rel) / np.arange(1, k + 1)
    return float((precisions * rel).sum() / rel.sum())


def evaluate(query_emb, gallery_emb, query_lab, gallery_lab, ks=(5, 10)):
    sims = query_emb @ gallery_emb.T  # cosine: embeddings are L2-normalised
    ranked = np.argsort(-sims, axis=1)
    out = {f"recall@{k}": [] for k in ks}
    out["map@10"] = []
    for i, q in enumerate(query_lab):
        labs = gallery_lab[ranked[i]]
        for k in ks:
            out[f"recall@{k}"].append(recall_at_k(labs, q, k))
        out["map@10"].append(ap_at_k(labs, q, 10))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}, ranked, sims


def per_class_recall(ranked, query_lab, gallery_lab, k=10):
    res = {}
    for c in sorted(set(query_lab.tolist())):
        idx = np.where(query_lab == c)[0]
        hits = [np.any(gallery_lab[ranked[i]][:k] == c) for i in idx]
        res[int(c)] = round(float(np.mean(hits)), 4)
    return res


def make_examples(data, ranked, n_show=6):
    """Save a montage: each row = one query + its top-5 retrieved images."""
    qimg, gimg = data["query_images"], data["gallery_images"]
    names = [str(x) for x in data["class_names"]]
    rows = []
    for qi in range(n_show):
        row = [qimg[qi]]
        row += [gimg[j] for j in ranked[qi][:5]]
        rows.append(row)
    cell = 64
    canvas = Image.new("RGB", (6 * cell, n_show * cell), "white")
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            canvas.paste(Image.fromarray(img).resize((cell, cell), Image.NEAREST),
                         (c * cell, r * cell))
    canvas.save(os.path.join(ROOT, "results", "examples.png"))
    # Demo HTML with the same examples, labelled.
    html = ["<html><head><meta charset='utf-8'><title>Visual search examples</title>"
            "<style>body{font-family:sans-serif}img{width:96px;height:96px;image-rendering:pixelated}"
            ".q{outline:3px solid #2563eb}</style></head><body>",
            "<h1>Visual search — example queries (CIFAR-10)</h1>",
            "<p>First image in each row is the query (blue outline); the next five are the top-5 retrieved images.</p>"]
    for qi in range(n_show):
        qlab = names[int(data["query_labels"][qi])]
        html.append(f"<h3>Query {qi + 1}: {qlab}</h3><div>")
        for j in [None] + list(ranked[qi][:5]):
            img = qimg[qi] if j is None else gimg[j]
            lab = qlab if j is None else names[int(data["gallery_labels"][j])]
            cls = " class='q'" if j is None else ""
            import base64, io
            buf = io.BytesIO()
            Image.fromarray(img).resize((96, 96), Image.NEAREST).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            html.append(f"<figure style='display:inline-block;margin:4px;text-align:center'>"
                        f"<img src='data:image/png;base64,{b64}'{cls}><figcaption>{lab}</figcaption></figure>")
        html.append("</div>")
    html.append("</body></html>")
    with open(os.path.join(ROOT, "results", "demo.html"), "w") as f:
        f.write("\n".join(html))


def main():
    data = np.load(NPZ, allow_pickle=True)
    gallery_emb, query_emb = data["gallery_emb"], data["query_emb"]
    gallery_lab, query_lab = data["gallery_labels"], data["query_labels"]
    names = [str(x) for x in data["class_names"]]

    resnet_metrics, ranked, _ = evaluate(query_emb, gallery_emb, query_lab, gallery_lab)
    hist_metrics, _, _ = evaluate(data["query_hist"], data["gallery_hist"],
                                  query_lab, gallery_lab)
    per_class = {names[c]: v for c, v in
                 per_class_recall(ranked, query_lab, gallery_lab).items()}

    results = {
        "gallery_size": int(len(gallery_lab)),
        "n_queries": int(len(query_lab)),
        "embedding_dim": int(gallery_emb.shape[1]),
        "model": "ResNet-18 pretrained on ImageNet (ONNX Model Zoo), global-average-pooling embedding, cosine similarity",
        "baseline": "64-d RGB colour histogram, cosine similarity",
        "resnet18": resnet_metrics,
        "colour_histogram_baseline": hist_metrics,
        "resnet18_recall@10_by_class": per_class,
        "relevance_definition": "retrieved image is relevant iff its CIFAR-10 class matches the query class (proxy for visual similarity)",
    }
    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    make_examples(data, ranked)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
