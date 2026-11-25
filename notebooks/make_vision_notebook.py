"""Write notebooks/TradeMesh_Vision_Colab.ipynb (the full-corpus page-image run for Google Colab).

    python notebooks/make_vision_notebook.py

The notebook is generated from this file so that it stays readable in review; open the .ipynb in Colab (File > Upload notebook, or the
GitHub tab) and choose Runtime > Change runtime type > T4 GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "TradeMesh_Vision_Colab.ipynb"
cells = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# TradeMesh: page-image retrieval over all 20 filings (Colab)

This notebook runs the **full-corpus version** of the page-image search that the app's **Pages** tab shows. It needs a GPU, so it is meant for Colab (free T4).

| Step | What happens | Roughly how long (free T4) |
|---|---|---|
| 1 | Download the latest 10-K of each of the 20 companies from SEC EDGAR, render every page to an image, extract each page's text, write the questions | 30 to 60 minutes (the rendering is the slow part) |
| 2 | Encode every page image and every question with a ColPali-family model (`colqwen2` by default) | 40 to 90 minutes |
| 3 | Rank all pages for every question with **our own MaxSim**, and with **BM25 over the page text** as a baseline; write `results.json` and small page thumbnails | 1 to 3 minutes |
| 4 | Download a small result bundle (or push it to GitHub) so the Pages tab shows the full run | 1 minute |

Everything is **resumable**: it stores its work in a Google Drive folder, so if Colab disconnects, run the cells again and it carries on where it stopped.

**What you need:** a Google account (Drive), the SEC contact string `Your Name your.address@example.com` stored as a Colab **Secret** named `SEC_USER_AGENT` (the key icon in the left bar; the SEC requires a contact address in the request header). Optional, only to push to GitHub: a Secret named `GITHUB_TOKEN` with write access to the repository. **Never paste a key or token into a cell.**

**Honest notes.** The pre-trained model is a component; the MaxSim scoring and the text baseline are the project's own code (`src/trademesh/vision/maxsim.py`, `src/trademesh/ranking/bm25.py`). A page counts as correct for a question when its text matches that question's keyword rule, which favours the text baseline a little. The filings' charts are separate image files that are not downloaded, so a chart shows as an empty frame on its page. This notebook has not been run on a GPU by its author; the first cells check each step, and the smoke test in step 2 stops early if the model does not load or gives bad vectors.
""")

md("## 0. Settings")
code('''
#@title Settings (edit, then run)
REPO_URL = "https://github.com/Deltasthicc/IRFinalProj"  #@param {type:"string"}
BRANCH = "main"  #@param {type:"string"}
#@markdown Folder in your Google Drive where the work is kept (so a disconnect does not lose it).
DRIVE_FOLDER = "TradeMesh_vision"  #@param {type:"string"}
MODEL = "colqwen2"  #@param ["colqwen2", "colpali", "colsmol-500m", "colsmol-256m"]
#@markdown Pages per batch on the GPU (lower it if you see an out-of-memory error).
BATCH = 2  #@param {type:"integer"}
#@markdown Leave empty for all 20 companies, or list tickers, for example AAPL,MSFT,TSLA,WMT.
TICKERS = ""  #@param {type:"string"}
import os, re
IN_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")
REPO_DIR = "/content/IRFinalProj"
DRIVE_FOLDER = re.sub(r"[^A-Za-z0-9_-]", "", DRIVE_FOLDER) or "TradeMesh_vision"
WORK = f"/content/drive/MyDrive/{DRIVE_FOLDER}"
print("model:", MODEL, "| work folder:", WORK, "| tickers:", TICKERS or "all 20")
''')

md("## 1. Setup and checks")
SCRIPTS = ["vision_full_prepare", "vision_full_encode", "vision_full_score"]
code('''
#@title GPU check
import subprocess
out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print(out.stdout.strip() or "NO GPU FOUND. Use Runtime > Change runtime type > T4 GPU, then run this cell again.")
assert out.stdout.strip(), "A GPU is needed for step 2 (encoding)."
''')
code('''
#@title Mount Google Drive and create the work folder
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(WORK, exist_ok=True)
print("work folder ready:", WORK, "| files already there:", len(os.listdir(WORK)))
''')
code('''
#@title Get the project code from GitHub
import subprocess, sys
if not os.path.exists(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, REPO_DIR], check=True)
else:
    print(subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], capture_output=True, text=True).stdout)
os.chdir(REPO_DIR)
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
assert os.path.exists("data/filings.json") and os.path.exists("src/trademesh/ranking/bm25.py"), "The project code is incomplete: check REPO_URL and BRANCH."
''')
md("""
The three scripts below are **part of this notebook** (written into the cloned project), so the notebook works even if the scripts are not on GitHub yet. They are the same files as `eval/vision_full_*.py` in the project.
""")
for name in SCRIPTS:
    body = (Path(__file__).resolve().parents[1] / "eval" / f"{name}.py").read_text(encoding="utf-8")
    code(f"%%writefile /content/IRFinalProj/eval/{name}.py\n" + body)
code('''
#@title Install what is needed (3 to 6 minutes the first time)
!pip install -q colpali-engine playwright pillow numpy
!playwright install --with-deps chromium > /dev/null 2>&1
!apt-get -qq install -y poppler-utils > /dev/null 2>&1
import torch, importlib.metadata as md
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| colpali-engine", md.version("colpali-engine"), "| transformers", md.version("transformers"))
!pdftoppm -v 2>&1 | head -1
assert torch.cuda.is_available()
''')
code('''
#@title SEC contact (from Colab Secrets; never printed or saved)
import getpass
try:
    from google.colab import userdata
    ua = userdata.get("SEC_USER_AGENT")
except Exception:
    ua = ""
if not ua or "@" not in ua:
    ua = getpass.getpass("SEC contact, as: Your Name your.address@example.com : ")
assert "@" in ua, "The SEC requires a contact address in the User-Agent."
os.environ["SEC_USER_AGENT"] = ua
print("SEC contact set (not shown).")
''')

md("""
## 2. Step 1: download, render and prepare the pages
Downloads each filing once (cached), prints each company as it is rendered. If it stops, run the cell again.
""")
code('''
#@title Step 1: prepare (download, render, extract text, write questions)
tick = f"--tickers {TICKERS}" if TICKERS.strip() else ""
!cd {REPO_DIR} && python eval/vision_full_prepare.py --work "{WORK}" {tick}
''')

md("""
## 3. Step 2: encode with the vision model
First a **smoke test** on three pages (it checks that the model loads and produces finite vectors), then the full encode. Both write to Drive, so the encode is resumable.
""")
code('''
#@title Smoke test: three pages
!cd {REPO_DIR} && python eval/vision_full_encode.py --work "{WORK}" --model {MODEL} --batch {BATCH} --limit 3
''')
code('''
#@title Step 2: encode every page and every question (the long step; safe to run again after a disconnect)
!cd {REPO_DIR} && python eval/vision_full_encode.py --work "{WORK}" --model {MODEL} --batch {BATCH}
''')

md("""
## 4. Step 3: score with our MaxSim and the text baseline
""")
code('''
#@title Step 3: rank all pages for every question
!cd {REPO_DIR} && python eval/vision_full_score.py --work "{WORK}" --out {REPO_DIR}/data/vision_full
''')
code('''
#@title Look at the result
import json
from IPython.display import display, Image as IPImage
res = json.load(open(f"{REPO_DIR}/data/vision_full/results.json"))
s = res["summary"]["overall"]
n = s["vision_maxsim"]["n"]
print(f'{n} questions over {res["n_pages"]} pages of {res["n_companies"]} companies')
for k in ("vision_maxsim", "text_bm25"):
    print(f'  {k:14s} first result correct {s[k]["hit@1"]}/{n}, in top 3 {s[k]["hit@3"]}/{n}, in top 10 {s[k]["hit@10"]}/{n}, MRR {s[k]["mrr"]}')
for g, v in res["summary"]["groups"].items():
    print(f'  {g:17s} vision {v["vision_maxsim"]["hit@1"]}/{v["vision_maxsim"]["n"]}   text {v["text_bm25"]["hit@1"]}/{v["text_bm25"]["n"]}')
q = res["queries"][0]
print("\\nExample:", q["query"])
for r in q["results"][:2]:
    print(r["ticker"], "page", r["page"], "score", round(r["score"], 2), "(correct)" if r["correct"] else "")
    display(IPImage(f'{REPO_DIR}/{r["image"]}', width=300))
''')

md("""
## 5. Get the result into the project
The result holds only what the Pages tab needs (`results.json`, about 200 page thumbnails and the evaluation file), a few megabytes. There are two ways to bring it to your laptop:

* **Push to GitHub (recommended):** the cell after the bundle pushes the result to a new branch (`vision-full-results`), using the `GITHUB_TOKEN` Secret. On your laptop, in the project folder, run `git fetch origin` and then `git checkout origin/vision-full-results -- data/vision_full eval/out/vision_full_eval.json`. Your own uncommitted work is left alone.
* **Download the bundle:** unzip it into the project folder on your laptop (it keeps the folder names).

Either way, start the app and open **Pages**: the app uses the full run automatically when `data/vision_full/results.json` exists.
""")
code('''
#@title Make the result bundle (download and copy to Drive)
import shutil, zipfile
bundle = "/content/vision_full_bundle.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(f"{REPO_DIR}/data/vision_full"):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, REPO_DIR))
    z.write(f"{REPO_DIR}/eval/out/vision_full_eval.json", "eval/out/vision_full_eval.json")
shutil.copy(bundle, f"{WORK}/vision_full_bundle.zip")
print("bundle:", round(os.path.getsize(bundle) / 1e6, 1), "MB (also saved in your Drive folder)")
from google.colab import files
files.download(bundle)
''')
code('''
#@title Push the result to a new GitHub branch (needs a GITHUB_TOKEN secret with write access)
PUSH = True  #@param {type:"boolean"}
NEW_BRANCH = "vision-full-results"  #@param {type:"string"}
GIT_NAME = "TradeMesh vision run"  #@param {type:"string"}
if PUSH:
    from google.colab import userdata
    try:
        token = userdata.get("GITHUB_TOKEN")
    except Exception:
        token = ""
    if not token:
        print("No GITHUB_TOKEN secret found, so nothing was pushed. Use the downloaded bundle above, or add the secret "
              "(a GitHub token with write access to the repository) and run this cell again.")
        PUSH = False
if PUSH:
    owner_repo = REPO_URL.rstrip("/").removesuffix(".git").split("github.com/")[1]
    def git(*a):
        r = subprocess.run(["git", *a], cwd=REPO_DIR, capture_output=True, text=True)
        print((r.stdout + r.stderr).replace(token, "***").strip())
        return r.returncode
    git("config", "user.name", GIT_NAME)
    git("config", "user.email", "noreply@example.com")
    git("checkout", "-B", NEW_BRANCH)
    git("add", "data/vision_full", "eval/out/vision_full_eval.json")
    git("commit", "-m", "Add the full-corpus page-image results (Colab run)")
    if git("push", f"https://x-access-token:{token}@github.com/{owner_repo}.git", NEW_BRANCH) == 0:
        print("Pushed to branch", NEW_BRANCH, "of", owner_repo)
        print("On your laptop, in the project folder:")
        print("  git fetch origin")
        print(f"  git checkout origin/{NEW_BRANCH} -- data/vision_full eval/out/vision_full_eval.json")
        print("then restart the app and open the Pages tab.")
else:
    print("Not pushed. Use the downloaded bundle instead.")
''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, len(cells), "cells")
