"""Typeset the executed JAX notebook as a compact 4:3 e-ink PDF."""

from __future__ import annotations

import base64
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = ROOT / "README.ipynb"
OUT = REPO / "tmp" / "pdfs" / "jax_full"
FINAL = ROOT / "README.pdf"
for directory in (OUT, OUT / "cells", OUT / "figures", FINAL.parent):
    directory.mkdir(parents=True, exist_ok=True)


PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=180mm,paperheight=240mm,left=10mm,right=10mm,
 top=12mm,bottom=12mm,headheight=11pt,headsep=3mm,footskip=6mm]{geometry}
\usepackage{fontspec}
\setmainfont{texgyrepagella}[Extension=.otf,UprightFont=*-regular,
 BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setsansfont{texgyreheros}[Extension=.otf,UprightFont=*-regular,
 BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\setmonofont{Inconsolatazi4}[Extension=.otf,UprightFont=*-Regular,
 BoldFont=*-Bold,Scale=.94]
\usepackage{amsmath,amssymb,unicode-math}
\setmathfont{texgyrepagella-math.otf}
\usepackage{microtype,xcolor,graphicx,adjustbox,longtable,booktabs,array,calc}
\usepackage{fvextra,upquote,enumitem,titlesec,needspace,etoolbox}
\usepackage{fancyhdr,lastpage,xurl}
\usepackage[hidelinks,unicode,pdfencoding=auto]{hyperref}
\usepackage{bookmark}
\hypersetup{pdftitle={Bayesian backpropagation in JAX},pdfauthor={},
 pdfsubject={Native JAX moment messages, implementation, and evidence}}
\definecolor{quiet}{gray}{0.32}
\setlength{\parindent}{0pt}\setlength{\parskip}{3pt plus .6pt minus .4pt}
\AtBeginDocument{\fontsize{10}{12.2}\selectfont}
\setlength{\emergencystretch}{4em}
\widowpenalty=10000\clubpenalty=10000\brokenpenalty=10000
\setlength{\abovedisplayskip}{4pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{4pt plus 1pt minus 1pt}
\allowdisplaybreaks[2]
\setlist{nosep,leftmargin=1.5em,topsep=2pt,parsep=1pt}
\providecommand{\tightlist}{\setlength{\itemsep}{1pt}\setlength{\parskip}{0pt}}
\setcounter{secnumdepth}{0}\setcounter{tocdepth}{1}
\titleformat{\section}{\sffamily\bfseries\fontsize{13}{15}\selectfont\raggedright}{}{0pt}{}
\titleformat{\subsection}{\sffamily\bfseries\fontsize{11}{13}\selectfont\raggedright}{}{0pt}{}
\titleformat{\subsubsection}{\bfseries\fontsize{10.2}{12}\selectfont\raggedright}{}{0pt}{}
\titlespacing*{\section}{0pt}{10pt plus 2pt}{4pt}
\titlespacing*{\subsection}{0pt}{7pt plus 1pt}{3pt}
\titlespacing*{\subsubsection}{0pt}{5pt plus 1pt}{2pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\fontsize{8}{9}\selectfont\sffamily\color{quiet}Bayesian backpropagation}
\fancyhead[R]{\fontsize{8}{9}\selectfont\sffamily\color{quiet}JAX implementation and evidence}
\fancyfoot[C]{\fontsize{8.5}{9.5}\selectfont\thepage\ / \pageref*{LastPage}}
\renewcommand{\headrulewidth}{0.25pt}\raggedbottom
\fvset{fontsize=\fontsize{9.2}{10.6}\selectfont,breaklines=true,
 breakanywhere=false,breakafter={,;/)]=},breakbefore={+},breakautoindent=false,
 breakindent=1em,breaksymbolleft={\tiny\ensuremath{\hookrightarrow}},
 breaksymbolright={},tabsize=2,baselinestretch=1,showspaces=false,showtabs=false,
 numbersep=5pt,xleftmargin=3mm,framesep=0pt}
\DefineVerbatimEnvironment{NotebookVerbatim}{Verbatim}{}
\newcommand{\CodeCell}[2]{\par\Needspace{4\baselineskip}
 {\fontsize{8.2}{9.5}\selectfont\sffamily\color{quiet}\textbf{Code cell #1}\quad #2}\par\vspace{-1pt}}
\newcommand{\OutputLabel}[1]{\par\Needspace{3\baselineskip}
 {\fontsize{8.2}{9.5}\selectfont\sffamily\color{quiet}#1}\par\vspace{-1pt}}
\newcommand{\NotebookFigure}[2]{\par\begin{center}\begin{minipage}{\linewidth}\centering
 \includegraphics[width=.95\linewidth,height=120mm,keepaspectratio]{#1}\par
 {\fontsize{8.2}{9.5}\selectfont\sffamily\color{quiet}#2}\end{minipage}\end{center}}
\setlength{\tabcolsep}{3pt}\setlength{\LTpre}{3pt}\setlength{\LTpost}{4pt}
\renewcommand{\arraystretch}{1.12}\AtBeginEnvironment{longtable}{\fontsize{9}{10.7}\selectfont}
\newcommand{\pandocbounded}[1]{\begin{adjustbox}{max width=\linewidth}#1\end{adjustbox}}
\begin{document}
{\fontsize{15}{18}\selectfont\sffamily\bfseries Bayesian backpropagation in JAX}\par
{\fontsize{10}{12}\selectfont\sffamily Native moment messages, implementation, and evidence}\par
"""


def raw_tex(text: str) -> str:
    return "\n\n```{=latex}\n" + text + "\n```\n\n"


def joined(value) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def tex_text(text: str) -> str:
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
               "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}",
               "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(char, char) for char in text).replace("\n", " ")


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.table, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table": self.table = []
        elif tag == "tr" and self.table is not None: self.row = []
        elif tag in {"td", "th"} and self.row is not None: self.cell = []

    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split())); self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row: self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            if self.table: self.tables.append(self.table)
            self.table = None


def table_tex(rows: list[list[str]], number: int) -> str:
    # Pandas' anonymous row index is not an experimental variable. Omitting
    # it gives the actual method/metric columns room on the compact page.
    if rows[0][0] == "" and all(row[0].isdigit() for row in rows[1:]):
        rows = [row[1:] for row in rows]
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    header, body = rows[0], rows[1:]
    identifiers = [0]
    if width > 1 and header[1] in {"optimizer", "rank", "quantity", "training"}:
        identifiers.append(1)
    if width > 2 and header[2] == "prediction":
        identifiers.append(2)
    other = [i for i in range(width) if i not in identifiers]
    capacity = max(2, 6 - len(identifiers))
    panels = [identifiers + other[i:i + capacity] for i in range(0, len(other), capacity)] or [identifiers]
    pieces = []
    for panel_number, columns in enumerate(panels, 1):
        n = len(columns)
        weights = []
        for column in columns:
            longest = max(len(row[column]) for row in rows)
            weights.append(min(2.7, max(1.0, longest / 13)))
        total = sum(weights)
        spec = "@{}" + "".join(
            r">{\raggedright\arraybackslash}p{(\linewidth - " + str(2 * (n - 1))
            + r"\tabcolsep) * \real{" + f"{weight / total:.6f}" + "}}"
            for weight in weights
        ) + "@{}"
        title = f"Table {number}" + (f" - panel {panel_number}/{len(panels)}" if len(panels) > 1 else "")
        pieces += [r"\begin{longtable}{" + spec + "}",
                   r"\multicolumn{" + str(n) + r"}{@{}l}{\sffamily\bfseries " + tex_text(title) + r"}\\*[3pt]",
                   r"\toprule",
                   " & ".join(r"\textbf{" + tex_text(header[i]) + "}" for i in columns) + r" \\",
                   r"\midrule\endfirsthead",
                   r"\toprule",
                   " & ".join(r"\textbf{" + tex_text(header[i]) + "}" for i in columns) + r" \\",
                   r"\midrule\endhead\bottomrule\endfoot"]
        pieces += [" & ".join(tex_text(row[i]) for i in columns) + r" \\" for row in body]
        pieces.append(r"\end{longtable}")
    return "\n".join(pieces)


def code_ascii(text):
    replacements={'ᵀ':'^T','⁻':'-','¹':'1','²':'^2','³':'^3','×':' x ','→':' -> ','≤':'<=','≥':'>=','ε':'epsilon','Σ':'Sigma','Λ':'Lambda','Φ':'Phi','μ':'mu','ρ':'rho','η':'eta','φ':'phi','π':'pi','∈':' in ','∇':'grad ','δ':'delta','α':'alpha','β':'beta','θ':'theta','≈':'~','∞':'infinity','…':'...','−':'-'}
    for a,b in replacements.items():text=text.replace(a,b)
    return ''.join(('|' if c in '│┃' else '-' if c in '─━' else '+') if 0x2500<=ord(c)<=0x257f else c for c in text)


def main():
    notebook_bytes = SOURCE.read_bytes()
    notebook = json.loads(notebook_bytes)
    chunks, figure_number, table_number = [], 0, 0
    for index, cell in enumerate(notebook["cells"], 1):
        source = joined(cell.get("source", ""))
        chunks.append(raw_tex(r"\hypertarget{cell-" + str(index) + "}{}"))
        if cell["cell_type"] == "markdown":
            if "support-source" in cell.get("metadata", {}).get("tags", []):
                label = cell["metadata"].get("source_label", "Supporting source")
                chunks.append(raw_tex(r"{\small\sffamily\color{quiet}" + tex_text(label)
                    + ". Expand notebook/HTML cell " + str(index) + r" for the complete source.}\par"))
                continue
            source = source.replace("\u2013", "-").replace("\u2014", "-").replace("\u2011", "-")
            if index == 1:
                # The separately typeset title is already in the preamble.
                source = re.sub(r'^# [^\n]+\n', '', source).lstrip()
            # Notebook chapters are H2 below its title; promote them for the
            # PDF's section-level page guide and chapter typography.
            source = re.sub(r'(?m)^## ', '# ', source)
            source=re.sub(r'```python\n(.*?)\n```',lambda m:'```python\n'+code_ascii(m[1])+'\n```',source,flags=re.S)
            chunks.append(source)
            if index == 1:
                chunks.append(raw_tex(r"\renewcommand{\contentsname}{Page guide}\tableofcontents\medskip"))
        else:
            filename = f"cells/cell_{index:03d}.py"
            (OUT / filename).write_text(code_ascii(source))
            if not set(cell.get("metadata", {}).get("tags", [])) & {"hide-input", "collapse-input"}:
                label = cell.get("metadata", {}).get("code_label", "Python / JAX")
                chunks.append(raw_tex(r"\CodeCell{" + str(index) + "}{" + tex_text(label) + "}\n"
                                      + r"\VerbatimInput[numbers=left]{" + filename + "}"))
        for output_index, output in enumerate(cell.get("outputs", []), 1):
            data = output.get("data", {})
            if "image/png" in data:
                figure_number += 1
                filename = f"figures/cell_{index:03d}_output_{output_index:03d}.png"
                (OUT / filename).write_bytes(base64.b64decode(joined(data["image/png"])))
                chunks.append(raw_tex(r"\NotebookFigure{" + filename + "}{Figure "
                                      + str(figure_number) + "}"))
            if "text/html" in data:
                parser = TableParser(); parser.feed(joined(data["text/html"]))
                for table in parser.tables:
                    table_number += 1
                    chunks.append(raw_tex(table_tex(table, table_number)))
                if not parser.tables:
                    from bs4 import BeautifulSoup
                    rich=BeautifulSoup(joined(data['text/html']),'html.parser')
                    for heading in rich.find_all(['h2','h3']):
                        chunks.append('### '+heading.get_text(' ',strip=True))
                    for link in rich.find_all('a',href=True):
                        href=link.get('data-reader-file',link['href'])
                        if href.startswith(('assets/','results/')):
                            chunks.append('['+rich.get_text(' ',strip=True)+']('+href+')')
            # Rich notebook values such as DataFrames carry both HTML and a
            # plain-text fallback. The table above is the reader-facing form;
            # do not print the same result twice.
            if output.get("output_type") == "stream" or (
                "text/plain" in data
                and "text/html" not in data
                and "image/png" not in data
            ):
                text = joined(output.get("text", "") if output.get("output_type") == "stream"
                              else data["text/plain"])
                text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "")
                text = "\n".join(
                    line for line in text.splitlines() if not line.startswith("[torch.onnx]")
                ).replace("✅", "OK")
                text = text.translate(str.maketrans({"├": "+", "└": "+", "─": "-", "│": "|"}))
                # Flax's decorative rules need not retain terminal width
                # on a narrow page; preserve all actual layer rows verbatim.
                text = re.sub(r"(?m)^([=\-])\1{95,}$", lambda m: m[1] * 95, text)
                filename = f"cells/cell_{index:03d}_output_{output_index:03d}.txt"
                (OUT / filename).write_text(code_ascii(text))
                chunks.append(raw_tex(r"\OutputLabel{Output - cell " + str(index) + "}\n"
                                      + r"\VerbatimInput[numbers=none,xleftmargin=0pt]{" + filename + "}"))

    markdown = "\n\n".join(chunks)
    body = subprocess.run(
        ["pandoc", "-f", "markdown+tex_math_dollars+tex_math_single_backslash",
         "-t", "latex", "--wrap=none", "--no-highlight"],
        input=markdown, text=True, capture_output=True, check=True,
    ).stdout
    body = body.replace(r"\begin{verbatim}", r"\begin{NotebookVerbatim}").replace(
        r"\end{verbatim}", r"\end{NotebookVerbatim}")
    body = re.sub(
        r"\\texttt\{([^{}]*)\}",
        lambda match: r"\texttt{" + match[1].replace(r"\_", r"\_\allowbreak{}")
        .replace("/", r"/\allowbreak{}").replace(r"\ ", r"\ \allowbreak{}") + "}",
        body,
    )
    (OUT / "tutorial.tex").write_text(PREAMBLE + body + "\n\\end{document}\n")
    subprocess.run(
        ["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "tutorial.tex"],
        cwd=OUT, check=True,
    )
    FINAL.write_bytes((OUT / "tutorial.pdf").read_bytes())
    receipt = {
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "figures": figure_number,
        "tables": table_number,
        "page_mm": [180, 240],
        "body_pt": 10,
        "code_pt": 9.2,
        "output": str(FINAL),
    }
    (OUT / "source_manifest.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
