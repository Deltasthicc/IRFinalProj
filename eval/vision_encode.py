"""Step 2 of the small page-image demo: turn the page images and the questions into multi-vectors with a ColPali-family model.

    .venv-vision\\Scripts\\python.exe eval/vision_encode.py [--model vidore/colSmol-256M] [--device cpu|cuda]

This is the only step that needs a vision-language model (the pre-trained encoder is used as a component; the scoring is ours,
in trademesh.vision.maxsim). It runs in its own virtual environment (.venv-vision: colpali-engine, torch, transformers) so that the
main environment is untouched. Writes data/vision/emb_pages.npz and data/vision/emb_queries.npz (one array per page / question)
and data/vision/encode_info.json (model, device, timings).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = ROOT / "data" / "vision"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vidore/colSmol-256M")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N pages (a speed test)")
    ap.add_argument("--queries-only", action="store_true", help="re-encode the questions only (after editing questions.json); keeps the page vectors")
    args = ap.parse_args()
    import torch
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    from PIL import Image

    pages = json.loads((V / "pages.json").read_text(encoding="utf-8"))
    questions = json.loads((V / "questions.json").read_text(encoding="utf-8"))
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    t0 = time.time()
    model = ColIdefics3.from_pretrained(args.model, torch_dtype=dtype, device_map=args.device).eval()
    processor = ColIdefics3Processor.from_pretrained(args.model)
    load_s = time.time() - t0
    print(f"model {args.model} loaded in {load_s:.0f}s on {args.device}", flush=True)

    ids = [] if args.queries_only else list(pages)[: args.limit or None]
    page_vecs, page_secs = {}, []
    for k, pid in enumerate(ids, 1):
        t1 = time.time()
        img = Image.open(ROOT / pages[pid]["image"]).convert("RGB")
        batch = processor.process_images([img]).to(model.device)
        with torch.no_grad():
            emb = model(**batch)[0].float().cpu().numpy()
        page_vecs[pid] = emb
        page_secs.append(time.time() - t1)
        print(f"  page {k}/{len(ids)} {pid}: {emb.shape[0]} vectors, {page_secs[-1]:.1f}s", flush=True)
    query_vecs = {}
    for q in questions:
        batch = processor.process_queries([q["question"]]).to(model.device)
        with torch.no_grad():
            query_vecs[q["id"]] = model(**batch)[0].float().cpu().numpy()
    if args.queries_only:
        np.savez_compressed(V / "emb_queries.npz", **query_vecs)
        print("re-encoded", len(query_vecs), "questions; page vectors unchanged")
    elif not args.limit:
        np.savez_compressed(V / "emb_pages.npz", **page_vecs)
        np.savez_compressed(V / "emb_queries.npz", **query_vecs)
        info = {"model": args.model, "device": args.device, "load_seconds": round(load_s, 1), "pages": len(ids),
                "seconds_per_page_mean": round(float(np.mean(page_secs)), 2), "vectors_per_page_mean": round(float(np.mean([v.shape[0] for v in page_vecs.values()])), 1),
                "vector_dim": int(next(iter(page_vecs.values())).shape[1]), "torch": torch.__version__}
        (V / "encode_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print("wrote embeddings;", info)


if __name__ == "__main__":
    main()
