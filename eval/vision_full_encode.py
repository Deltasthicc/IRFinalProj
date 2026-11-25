"""Full-corpus page-image run, step 2: encode every page image and every question with a ColPali-family model (needs a GPU).

    python eval/vision_full_encode.py --work /content/drive/MyDrive/TradeMesh_vision [--model colqwen2] [--batch 2]
    python eval/vision_full_encode.py --work data/vision_full_work --model colsmol-256m --limit 3     # a quick code-path test on a CPU

Reads <work>/pages.json and questions.json (from vision_full_prepare.py). Writes one float16 array per page to <work>/emb/<page id>.npy
(resumable: pages already encoded are skipped), the question vectors to <work>/emb_queries.npz and <work>/encode_info.json.
The pre-trained model is a component; the scoring is ours (trademesh.vision.maxsim, step 3).

--model: colqwen2 (vidore/colqwen2-v1.0-merged, default), colpali (vidore/colpali-v1.3-merged), colsmol-500m, colsmol-256m.
The "-merged" repositories contain the full weights, so no gated base model or Hugging Face login is needed.
Precision: bfloat16 on GPUs that support it, float16 otherwise; if a vector is not finite the run stops and tells you to rerun with
--dtype float32 (slower, needs about 9 GB for colqwen2).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "colqwen2": ("vidore/colqwen2-v1.0-merged", "ColQwen2", "ColQwen2Processor"),
    "colpali": ("vidore/colpali-v1.3-merged", "ColPali", "ColPaliProcessor"),
    "colsmol-500m": ("vidore/colSmol-500M", "ColIdefics3", "ColIdefics3Processor"),
    "colsmol-256m": ("vidore/colSmol-256M", "ColIdefics3", "ColIdefics3Processor"),
}


def to_vectors(out, batch_mask=None) -> list[np.ndarray]:
    """(B, T, D) model output -> one (n_i, D) float16 array per item, dropping padding rows (they are all zero)."""
    arr = out.float().cpu().numpy()
    res = []
    for item in arr:
        keep = np.linalg.norm(item, axis=1) > 0
        res.append(item[keep].astype(np.float16))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", default="colqwen2", choices=list(MODELS))
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--limit", type=int, default=0, help="encode only the first N pages (a code-path test); does not write the question vectors")
    args = ap.parse_args()
    import torch
    import colpali_engine.models as cm
    from PIL import Image

    work = Path(args.work).resolve()
    pages = json.loads((work / "pages.json").read_text(encoding="utf-8"))
    questions = json.loads((work / "questions.json").read_text(encoding="utf-8"))
    repo, model_cls, proc_cls = MODELS[args.model]
    cuda = torch.cuda.is_available()
    if args.dtype == "auto":
        dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if cuda else torch.float32
    else:
        dtype = getattr(torch, args.dtype)
    device = "cuda" if cuda else "cpu"
    t0 = time.time()
    model = getattr(cm, model_cls).from_pretrained(repo, torch_dtype=dtype, device_map=device).eval()
    processor = getattr(cm, proc_cls).from_pretrained(repo)
    print(f"{repo} on {device} ({dtype}), loaded in {time.time() - t0:.0f}s"
          + (f", GPU {torch.cuda.get_device_name(0)}" if cuda else ""), flush=True)

    emb = work / "emb"
    emb.mkdir(exist_ok=True)
    ids = list(pages)[: args.limit or None]
    todo = [pid for pid in ids if not (emb / f"{pid}.npy").exists()]
    print(f"{len(ids)} pages, {len(ids) - len(todo)} already encoded, {len(todo)} to do", flush=True)
    t1, done = time.time(), 0
    for i in range(0, len(todo), args.batch):
        chunk = todo[i:i + args.batch]
        images = [Image.open(work / pages[pid]["image"]).convert("RGB") for pid in chunk]
        batch = processor.process_images(images).to(model.device)
        with torch.no_grad():
            vecs = to_vectors(model(**batch))
        for pid, v in zip(chunk, vecs):
            if not np.isfinite(v.astype(np.float32)).all() or len(v) == 0:
                raise SystemExit(f"non-finite or empty vectors for {pid}; rerun with --dtype float32")
            np.save(emb / f"{pid}.npy", v)
        done += len(chunk)
        if done % 20 < args.batch or done == len(todo):
            rate = (time.time() - t1) / done
            print(f"  {done}/{len(todo)} pages, {rate:.2f} s/page, about {rate * (len(todo) - done) / 60:.0f} min left", flush=True)
    if args.limit:
        print("limited run: question vectors not written")
        return
    qvecs = {}
    for q in questions:
        batch = processor.process_queries([q["question"]]).to(model.device)
        with torch.no_grad():
            qvecs[q["id"]] = to_vectors(model(**batch))[0]
    np.savez_compressed(work / "emb_queries.npz", **qvecs)
    info = {"model": repo, "family": args.model, "device": device, "dtype": str(dtype).replace("torch.", ""), "pages": len(ids),
            "vector_dim": int(next(iter(qvecs.values())).shape[1]), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if cuda else None, "encode_seconds": round(time.time() - t0)}
    (work / "encode_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("wrote question vectors;", info)


if __name__ == "__main__":
    main()
