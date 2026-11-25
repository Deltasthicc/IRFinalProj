"""Build the presentation notes PDF from docs/presentation/PRESENTATION_NOTES.md.

    python docs/presentation/build_notes.py          # writes TradeMesh_Presentation_Notes.html and .pdf (needs Chrome or Edge)

Figure tokens in the markdown:
    @@FIG:path|width_mm|caption@@
    @@FIGROW:path1|x0,y0,x1,y1|path2|x0,y0,x1,y1|caption@@   (two screenshots side by side, each cropped)
"""
from __future__ import annotations

import base64
import html as H
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "TradeMesh_Presentation_Notes"
src = (HERE / "PRESENTATION_NOTES.md").read_text(encoding="utf-8")
title = re.search(r"^TITLE:\s*(.*)$", src, flags=re.M).group(1)
subtitle = re.search(r"^SUBTITLE:\s*(.*)$", src, flags=re.M).group(1)
body = re.sub(r"^(TITLE|SUBTITLE):.*\n", "", src, flags=re.M)


def uri(path: Path, box=None) -> str:
    if path.suffix == ".svg":
        return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()
    im = Image.open(path)
    if box:
        im = im.crop(box)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


figs: list[str] = []
count = 0


def stash(h: str) -> str:
    figs.append(h)
    return f"\n\nFIGPLACEHOLDER{len(figs) - 1}X\n\n"


def fig(m: re.Match) -> str:
    path, width, cap = m.group(1), m.group(2), m.group(3)
    p = ROOT / path
    if not p.exists():
        sys.exit(f"missing figure {path}")
    return stash(f'<figure><img style="max-width:{width}mm" src="{uri(p)}"><figcaption>{H.escape(cap)}</figcaption></figure>')


def figrow(m: re.Match) -> str:
    p1, b1, p2, b2, cap = m.groups()
    imgs = []
    for p, b in ((p1, b1), (p2, b2)):
        path = ROOT / p
        if not path.exists():
            sys.exit(f"missing figure {p}")
        imgs.append(f'<img src="{uri(path, tuple(int(v) for v in b.split(",")))}">')
    return stash(f'<figure><div class="row">{"".join(imgs)}</div><figcaption>{H.escape(cap)}</figcaption></figure>')


body = re.sub(r"@@FIGROW:([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|(.*?)@@", figrow, body)
body = re.sub(r"@@FIG:([^|]+)\|(\d+)\|(.*?)@@", fig, body)
html_body = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
html_body = re.sub(r"<p>FIGPLACEHOLDER(\d+)X</p>", lambda m: figs[int(m.group(1))], html_body)
toc = "".join(f"<li>{H.escape(h)}</li>" for h in re.findall(r"^## (.*)$", body, flags=re.M))
left = re.findall(r"@@\w+", html_body)
if left:
    sys.exit(f"unresolved tokens: {left[:3]}")

CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm; @bottom-center { content: counter(page); font: 8.5pt 'Segoe UI', Arial, sans-serif; color: #666; } }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1b1b1b; margin: 0; }
.cover { min-height: 235mm; padding-top: 22mm; }
.cover h1 { font-size: 30pt; line-height: 1.15; color: #1f3a5f; margin: 0 0 6mm; border: none; }
.cover .sub { font-size: 12.5pt; color: #444; max-width: 150mm; margin-bottom: 10mm; }
.cover .meta { font-size: 10pt; color: #333; margin-bottom: 9mm; line-height: 1.6; }
.cover h3 { font-size: 11pt; color: #1f3a5f; margin: 0 0 2mm; border: none; }
.cover ol { margin: 0; padding-left: 6mm; columns: 2; font-size: 10pt; line-height: 1.7; }
h1 { font-size: 18pt; }
h2 { font-size: 15pt; color: #1f3a5f; margin: 9mm 0 3mm; padding-bottom: 1.2mm; border-bottom: 1.4pt solid #1f3a5f; break-after: avoid; }
h3 { font-size: 11.5pt; color: #2d62b5; margin: 5.5mm 0 1.5mm; break-after: avoid; }
p { margin: 0 0 2.6mm; text-align: left; }
ul, ol { margin: 0 0 3mm; padding-left: 6mm; }
li { margin-bottom: 1mm; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 8.6pt; background: #f1f3f5; padding: 0 1mm; border-radius: 1mm; overflow-wrap: anywhere; }
pre { background: #f1f3f5; border-left: 2pt solid #2d62b5; padding: 2.5mm 3mm; font-size: 8.6pt; white-space: pre-wrap; break-inside: avoid; margin: 0 0 3mm; }
pre code { background: none; padding: 0; }
blockquote { margin: 2mm 0 3mm; padding: 2mm 4mm; background: #eef3fb; border-left: 3pt solid #2d62b5; font-size: 10.5pt; }
blockquote p { margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.7pt; line-height: 1.3; margin: 1mm 0 4mm; break-inside: auto; }
th, td { border: .5pt solid #b5bcc7; padding: 1.2mm 1.8mm; vertical-align: top; text-align: left; overflow-wrap: break-word; }
th { background: #e8edf5; color: #1f3a5f; }
tr { break-inside: avoid; }
tr:nth-child(even) td { background: #fafbfc; }
figure { margin: 2mm 0 4mm; text-align: center; break-inside: avoid; }
figure img { max-height: 112mm; }
.row { display: flex; justify-content: center; gap: 4mm; }
.row img { width: 78mm; max-height: none; }
figcaption { font-size: 8.6pt; color: #444; margin-top: 1.5mm; text-align: center; }
.callout { border-left: 3pt solid #888; background: #f6f7f9; padding: 2mm 3.2mm; margin: 1.5mm 0 3.2mm; font-size: 9.3pt; line-height: 1.4; break-inside: avoid; border-radius: 0 1.5mm 1.5mm 0; }
.callout.say { border-color: #2f7d4f; background: #eef7f1; }
.callout.demo { border-color: #d9822b; background: #fdf4e9; }
.callout.code { border-color: #2d62b5; background: #edf3fc; }
.tag { font-size: 8pt; font-weight: 700; padding: 0 1.4mm; border-radius: 1mm; color: #fff; }
.tag.say { background: #2f7d4f; } .tag.demo { background: #d9822b; } .tag.code { background: #2d62b5; }
"""
cover = (f'<div class="cover"><h1>{H.escape(title)}</h1><div class="sub">{H.escape(subtitle)}</div>'
         '<div class="meta"><b>Shashwat Rajan</b> (2510111019), <b>Tanish Shitanshu</b> (2510110008), <b>Ruchir Sharma</b> (2310110728)<br>'
         'CSD358 Information Retrieval, Monsoon 2026, Shiv Nadar Institution of Eminence (instructor: Sonia Khetarpaul)<br>'
         'Repository: https://github.com/Deltasthicc/IRFinalProj<br>Companion to the report <i>TradeMesh_Report.pdf</i>.</div>'
         f'<h3>Contents</h3><ol>{toc}</ol></div>')
page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{H.escape(title)}</title><style>{CSS}</style></head><body>{cover}{html_body}</body></html>"
OUT.with_suffix(".html").write_text(page, encoding="utf-8")
browser = next((p for p in [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            shutil.which("chrome") or "", shutil.which("msedge") or ""] if p and Path(p).exists()), None)
if browser:
    pdf = OUT.with_suffix(".pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", OUT.with_suffix(".html").as_uri()],
                   check=True, timeout=240)
    print("wrote", pdf, pdf.stat().st_size // 1024, "KB")
else:
    print("no Chrome/Edge found; open the .html and print to PDF")
