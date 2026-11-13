"""Typeset the executed notebook for Paper Pro; never execute or edit its cells.

Run with the project's Python (nbformat is not needed):
  python tensorflow/build_pdf.py

Markdown becomes real LaTeX through Pandoc. Code is included verbatim from
per-cell files. Output tables are split into readable column panels; every
displayed value and every embedded image is retained.
"""
from pathlib import Path
import base64
import os
import hashlib
import json
import re
import subprocess
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "README.ipynb"
OUT = ROOT.parent / "tmp" / "pdfs" / "tensorflow_full"
OUT.mkdir(parents=True, exist_ok=True)
for folder in ("cells", "figures"):
    (OUT / folder).mkdir(exist_ok=True)

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=180mm,paperheight=240mm,left=10mm,right=10mm,
 top=12mm,bottom=12mm,headheight=11pt,headsep=3mm,footskip=6mm]{geometry}
\usepackage{fontspec}
\setmainfont{texgyrepagella}[Extension=.otf,UprightFont=*-regular,
 BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,
 BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{Inconsolatazi4}[Extension=.otf,UprightFont=*-Regular,
 BoldFont=*-Bold,Scale=1.0]
\newfontfamily\NativeSummaryFont{Menlo}
\usepackage{amsmath,amssymb}
\usepackage{unicode-math}
\setmathfont{texgyrepagella-math.otf}
\usepackage{microtype}
\usepackage{xcolor,graphicx,adjustbox}
\usepackage{longtable,booktabs,array,calc}
\usepackage{fvextra,upquote}
\usepackage{enumitem,titlesec,needspace,etoolbox}
\usepackage{fancyhdr,lastpage}
\usepackage{xurl}
\usepackage[hidelinks,unicode,pdfencoding=auto]{hyperref}
\usepackage{bookmark}
\hypersetup{pdftitle={Bayesify: Implementation and experiments},
 pdfauthor={},pdfsubject={Implementation, mathematics, and recorded results}}
\definecolor{quiet}{gray}{0.32}
\definecolor{rulegray}{gray}{0.70}
\setlength{\parindent}{0pt}
\setlength{\parskip}{3.3pt plus .7pt minus .5pt}
\AtBeginDocument{\fontsize{10}{12.2}\selectfont}
\setlength{\emergencystretch}{4em}
\widowpenalty=10000\clubpenalty=10000\brokenpenalty=10000
\setlength{\abovedisplayskip}{5pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{5pt plus 1pt minus 1pt}
\setlength{\abovedisplayshortskip}{2pt}
\setlength{\belowdisplayshortskip}{4pt}
\allowdisplaybreaks[2]
\setlist{nosep,leftmargin=1.5em,topsep=3pt,parsep=2pt}
\providecommand{\tightlist}{\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{0}
\setcounter{tocdepth}{1}
\titleformat{\section}{\sffamily\bfseries\fontsize{13}{15}\selectfont\raggedright}{}{0pt}{}
\titleformat{\subsection}{\sffamily\bfseries\fontsize{11}{13}\selectfont\raggedright}{}{0pt}{}
\titleformat{\subsubsection}{\bfseries\fontsize{10.5}{12.5}\selectfont\raggedright}{}{0pt}{}
\titleformat{\paragraph}[block]{\bfseries\fontsize{10}{12.2}\selectfont}{}{0pt}{}
\titleformat{\subparagraph}[block]{\bfseries\fontsize{10}{12.2}\selectfont}{}{0pt}{}
\titlespacing*{\section}{0pt}{11pt plus 2pt}{5pt}
\titlespacing*{\subsection}{0pt}{8pt plus 1pt}{3pt}
\titlespacing*{\subsubsection}{0pt}{6pt plus 1pt}{2pt}
\titlespacing*{\paragraph}{0pt}{5pt}{2pt}
\titlespacing*{\subparagraph}{0pt}{5pt}{2pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\fontsize{8}{9}\selectfont\sffamily\color{quiet}Bayesify}
\fancyhead[R]{\fontsize{8}{9}\selectfont\sffamily\color{quiet}Implementation and experiments}
\fancyfoot[C]{\fontsize{9}{10}\selectfont\thepage\, / \,\pageref*{LastPage}}
\renewcommand{\headrulewidth}{0.25pt}
\renewcommand{\footrulewidth}{0pt}
\raggedbottom
\fvset{fontsize=\fontsize{9.2}{10.6}\selectfont,
 breaklines=true,breakanywhere=false,breakafter={,;/)]=},breakbefore={+},
 breakautoindent=false,breakindent=1em,
 breaksymbolleft={\tiny\ensuremath{\hookrightarrow}},breaksymbolright={},
 breakanywheresymbolpre={},breakanywheresymbolpost={},
 tabsize=2,baselinestretch=1,showspaces=false,showtabs=false,
 numbersep=5pt,xleftmargin=4mm,framesep=0pt}
\DefineVerbatimEnvironment{NotebookVerbatim}{Verbatim}{}
\newcommand{\CodeCell}[2]{\par\Needspace{4\baselineskip}
 {\fontsize{8.5}{10}\selectfont\sffamily\color{quiet}\textbf{Code cell #1}\quad #2}\par\vspace{-1pt}}
\newcommand{\OutputLabel}[1]{\par\Needspace{3\baselineskip}
 {\fontsize{8.5}{10}\selectfont\sffamily\color{quiet}#1}\par\vspace{-1pt}}
\newcommand{\NotebookFigure}[2]{\par\begin{center}\begin{minipage}{\linewidth}\centering\vspace{-3pt}
 \includegraphics[width=.96\linewidth,height=122mm,keepaspectratio]{#1}\par
 {\fontsize{8.3}{10}\selectfont\sffamily\color{quiet}#2}\vspace{-3pt}\end{minipage}\end{center}}
\newcommand{\NotebookFigureTall}[2]{\par\begin{center}\begin{minipage}{\linewidth}\centering\vspace{-3pt}
 \includegraphics[width=.96\linewidth,height=142mm,keepaspectratio]{#1}\par
 {\fontsize{8.3}{10}\selectfont\sffamily\color{quiet}#2}\vspace{-3pt}\end{minipage}\end{center}}
\newcommand{\NotebookArchitecture}[2]{\par\begin{center}\begin{minipage}{\linewidth}\centering\vspace{-3pt}
 \includegraphics[width=.90\linewidth,height=100mm,keepaspectratio]{#1}\par
 {\fontsize{8.3}{10}\selectfont\sffamily\color{quiet}#2}\vspace{-3pt}\end{minipage}\end{center}}
\setlength{\tabcolsep}{3pt}
\setlength{\LTpre}{4pt}\setlength{\LTpost}{5pt}
\renewcommand{\arraystretch}{1.14}
\AtBeginEnvironment{longtable}{\fontsize{9}{10.7}\selectfont}
\newcommand{\TablePanel}[1]{\par\Needspace{5\baselineskip}
 {\fontsize{9}{11}\selectfont\sffamily\bfseries #1}\par}
\newcommand{\pandocbounded}[1]{\begin{adjustbox}{max width=\linewidth}#1\end{adjustbox}}
\begin{document}
{\fontsize{15}{18}\selectfont\sffamily\bfseries Bayesify}\par
{\fontsize{10}{12}\selectfont\sffamily Implementation and experiments}\par
"""


def raw_tex(s):
    return "\n\n```{=latex}\n" + s + "\n```\n\n"


def tex_text(s):
    """Escape displayed text, adding invisible break opportunities in identifiers."""
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
               "$": r"\$", "#": r"\#", "_": r"\_\allowbreak{}",
               "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}
    return "".join(escapes.get(ch, ch) for ch in s).replace("\n", " ").replace("covariance", r"covari\-ance")


def output_table(table, table_number):
    """Repeat identifying columns; keep numeric cells at 9pt, never scale a table."""
    rows = [[c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"], recursive=False)]
            for tr in table.find_all("tr")]
    rows = [r for r in rows if r]
    if not rows:
        return "", 0
    width = max(map(len, rows))
    assert all(len(r) == width for r in rows), "Unexpected spanning output table"
    header, body = rows[0], rows[1:]
    quality_cost = width == 10 and "Accuracy" in header[1] and "Prediction" in header[-1]
    mnist_comparison = header[:3] == ["Model", "Dataset", "Method"]
    if quality_cost:
        panels = [[0,1,2,3,4,5],[0,6,7,8,9]]
    elif mnist_comparison:
        panels = [[0,1,2,3,4,5], [0,1,2,6,7,8], [0,1,2,9,10], [0,1,2,11,12,13,14]]
    elif header[:3] == ["Belief", "Model", "Method"]:
        panels = [list(range(width))]
    elif width <= 6:
        panels = [list(range(width))]
    else:
        identifiers = [0, 1] if header[1].lower() in {"seed", "dataset", "benchmark", "split"} else [0]
        other = [j for j in range(width) if j not in identifiers]
        capacity = 4 if len(identifiers) == 1 else 3
        panels = [identifiers + other[i:i + capacity] for i in range(0, len(other), capacity)]
    pieces = []
    for k, cols in enumerate(panels):
        # Long textual cells get more width; numeric columns keep space for long headers.
        weights = []
        for j in cols:
            strings = [r[j] for r in rows]
            longest = max(map(len, strings))
            numeric = all(re.fullmatch(r"[-+0-9.eE% ±×xNAnaInf /]+", r[j]) for r in body if r[j])
            weight = min(3.0, max(1.0, longest / 12)) if not numeric else 1.0
            if j == 0 and len(cols) > 3:
                weight = max(weight, 1.8)
            weights.append(weight)
        if mnist_comparison:
            # Short model names need little space; preserve 9pt numeric cells
            # and keep mean/SD pairs together wherever the page allows it.
            weights = [[.55,1,1.6,1.35,1.65,1.1],
                       [.55,1.15,1.6,1.65,1.8,1.55],
                       [.6,1.2,1.8,1.6,1.6],
                       [.5,1,1.5,1.25,1.4,1.1,1.1]][k]
        elif header[:3] == ["Belief", "Model", "Method"]:
            weights = [.8,.7,1.2,1.5,1.5,1.5,.6]
        total = sum(weights)
        n = len(cols)
        spec = "@{}" + "".join(
            r">{\raggedright\arraybackslash}p{(\linewidth - " + str(2 * (n - 1))
            + r"\tabcolsep) * \real{" + f"{w / total:.6f}" + "}}" for w in weights) + "@{}"
        panel = ("Predictive quality" if k == 0 else "Measured computation") if quality_cost else f"Table {table_number}"
        if len(panels) > 1 and not quality_cost:
            panel += f" · panel {k + 1}/{len(panels)}"
        if mnist_comparison:
            panel += " · " + ["Predictive quality", "Uncertainty and decisions",
                                "Fashion-MNIST detection", "Measured computation"][k]
        if mnist_comparison:
            pieces.append(r"\Needspace{" + str(len(body)+7) + r"\baselineskip}")
        elif len(body) <= 8:
            pieces.append(r"\Needspace{" + str(len(body)+5) + r"\baselineskip}")
        pieces.append(r"\begin{longtable}{" + spec + "}")
        title = (r"\multicolumn{" + str(n) + r"}{@{}l}{\sffamily\bfseries "
                 + tex_text(panel) + r"}\\*[4pt]")
        pieces.extend([title, r"\toprule"])
        h = " & ".join(r"\textbf{" + tex_text(header[j]) + "}" for j in cols) + r" \\"
        pieces.extend([h, r"\midrule", r"\endfirsthead", title, r"\toprule", h,
                       r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"])
        for row in body:
            pieces.append(" & ".join(tex_text(row[j]) for j in cols) + r" \\")
        pieces.append(r"\end{longtable}")
    return "\n".join(pieces), len(panels)


def main():
    source_bytes = SOURCE.read_bytes()
    nb = json.loads(source_bytes)
    chunks, cells, counts = [], [], {"code_cells": 0, "figures": 0, "tables": 0, "table_panels": 0,
                                   "streams": 0, "plain_outputs": 0, "markdown_cells": 0, "hidden_inputs": 0}
    for i, cell in enumerate(nb["cells"], 1):
        source = "".join(cell.get("source", []))
        chunks.append(raw_tex(r"\hypertarget{cell-" + str(i) + "}{}"))
        entry = {"cell": i, "type": cell["cell_type"],
                 "source_sha256": hashlib.sha256(source.encode()).hexdigest()}
        if cell["cell_type"] == "markdown":
            counts["markdown_cells"] += 1
            # HTML disclosure controls have no printed meaning; preserve their text.
            source = re.sub(r"<summary>(.*?)</summary>", r"**\1**\n", source, flags=re.S)
            source = re.sub(r"</?details>", "", source)
            chunks.append(source)
            if i == 1:
                chunks.append(raw_tex(r"\renewcommand{\contentsname}{Page guide}" + "\n"
                                      + r"\tableofcontents\medskip"))
        elif cell["cell_type"] == "code":
            counts["code_cells"] += 1
            filename = f"cells/cell_{i:03d}.R"
            (OUT / filename).write_text(source[4:] if source.startswith("%%R\n") else source)
            if "hide-input" in cell.get("metadata", {}).get("tags", []):
                counts["hidden_inputs"] += 1
            else:
                label=cell.get("metadata",{}).get("code_label","R / TensorFlow")
                assert label in ['R / TensorFlow','R / saved-result replay']
                chunks.append(raw_tex(r"\CodeCell{" + str(i) + "}{" + label + "}" + "\n"
                                      + r"\VerbatimInput[numbers=left]{" + filename + "}"))
            entry["code_file"] = filename
        else:
            raise ValueError(f"Unhandled cell type: {cell['cell_type']}")
        for j, output in enumerate(cell.get("outputs", []), 1):
            data = output.get("data", {})
            if "image/png" in data:
                counts["figures"] += 1
                filename = f"figures/cell_{i:03d}_output_{j:03d}.png"
                (OUT / filename).write_bytes(base64.b64decode("".join(data["image/png"])))
                figure_command = r"\NotebookArchitecture{" if (cell.get("id")=="roberta-lora-architecture" or (cell.get("id", "").startswith("native-") or "native_architecture(" in source)) else r"\NotebookFigureTall{" if cell.get("id") in ["gaussian-network-output","ep-message-agreement-output","mfvi-message-agreement-output"] else r"\NotebookFigure{"
                chunks.append(raw_tex(figure_command + filename + "}{"
                                      + f"Figure {counts['figures']}" + "}"))
            elif "text/html" in data:
                soup = BeautifulSoup("".join(data["text/html"]), "html.parser")
                for script in soup.find_all(["script", "style"]):
                    script.decompose()
                for node in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "table", "p"]):
                    if node.find_parent("table"):
                        continue
                    if node.name == "table":
                        counts["tables"] += 1
                        latex, panels = output_table(node, counts["tables"])
                        counts["table_panels"] += panels
                        chunks.append(raw_tex(latex))
                    elif node.name.startswith("h"):
                        chunks.append(raw_tex(r"\Needspace{9\baselineskip}\subsubsection*{"
                                              + tex_text(node.get_text(" ", strip=True)) + "}"))
                    else:
                        chunks.append(raw_tex(tex_text(node.get_text(" ", strip=True))))
            elif output.get("output_type") == "stream" or "text/plain" in data:
                is_stream = output.get("output_type") == "stream"
                text = "".join(output.get("text", []) if is_stream else data["text/plain"])
                counts["streams" if is_stream else "plain_outputs"] += 1
                # Strip terminal control codes, not the displayed text.
                text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "")
                filename = f"cells/cell_{i:03d}_output_{j:03d}.txt"
                (OUT / filename).write_text(text)
                native = cell.get("id")=="roberta-lora-architecture" or (cell.get("id", "").startswith("native-") or "native_architecture(" in source)
                summary = r"\OutputLabel{Recorded output · cell " + str(i) + "}\n" + r"\VerbatimInput[numbers=none,xleftmargin=0pt" + (r",formatcom=\NativeSummaryFont" if native else "") + r"]{" + filename + "}"
                if native:
                    summary = r"\begin{minipage}{\linewidth}" + "\n" + summary + "\n" + r"\end{minipage}"
                chunks.append(raw_tex(summary))

            elif output.get("output_type") == "error":
                raise ValueError(f"Notebook error output in cell {i}")
            elif data:
                raise ValueError(f"Unhandled output types: {list(data)}")
        cells.append(entry)

    markdown = "\n\n".join(chunks)
    body = subprocess.run(["pandoc", "-f", "markdown+tex_math_dollars+tex_math_single_backslash",
                           "-t", "latex", "--wrap=none", "--no-highlight"],
                          input=markdown, text=True, capture_output=True, check=True).stdout
    # Pandoc's identifiers differ from Jupyter's case-preserving fragment links.
    labels = re.findall(r"\\label\{([^}]+)\}", body)
    def norm(label):
        label = re.sub(r"ux[0-9a-f]{2}", "", label.lower())
        return re.sub(r"[^a-z0-9]", "", label.lower())
    label_map = {norm(s): s for s in labels}
    unresolved = []
    def relink(match):
        old = match.group(1)
        new = label_map.get(norm(old))
        if not new:
            unresolved.append(old)
            new = old
        return r"\hyperref[" + new + "]"
    body = re.sub(r"\\hyperref\[([^]]+)\]", relink, body)
    body = body.replace(r"\begin{verbatim}", r"\begin{NotebookVerbatim}").replace(
        r"\end{verbatim}", r"\end{NotebookVerbatim}")
    # Let long code identifiers wrap in prose and mapping tables without changing text.
    body = re.sub(r"\\texttt\{([^{}]*)\}",
                  lambda m: r"\texttt{" + m[1].replace(r"\_", r"\_\allowbreak{}")
                  .replace("(", r"(\allowbreak{}") + "}", body)
    body = body.replace("message/custom-gradient", r"message/\allowbreak{}custom-gradient")
    # Keep the short three-model comparison table together for the reader.
    body = body.replace(r"\subsection{Models used in the three comparisons}",
                        r"\Needspace{65mm}\subsection{Models used in the three comparisons}")
    (OUT / "tutorial.tex").write_text(PREAMBLE + "\n" + body + "\n\\end{document}\n")
    receipt = {"notebook": str(SOURCE), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
               "counts": counts, "cells": cells, "unresolved_links": unresolved,
               "page_mm": [180, 240], "body_pt": 10, "code_pt": 9.2,
               "method": "Notebook cells → Pandoc LaTeX → XeLaTeX; no notebook execution",
               "device_reference": "https://remarkable.com/products/remarkable-paper/pro/details/features"}
    (OUT / "source_manifest.json").write_text(json.dumps(receipt, indent=2))
    subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "tutorial.tex"], cwd=OUT, check=True)
    (ROOT / "README.pdf").write_bytes((OUT / "tutorial.pdf").read_bytes())
    print(json.dumps({"output": str(ROOT / "README.pdf"), "counts": counts, "unresolved_links": unresolved}, indent=2))


if __name__ == "__main__":
    main()
