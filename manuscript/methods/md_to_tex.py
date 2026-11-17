"""Turn the methods draft (methods_draft.md, exported from the Claude Doc) into LaTeX.

    methods.tex      main-text methods: Overview to Metadata extraction, Figures M1-M2, Tables M1-M3
    supplement.tex   Supplementary Notes 1-4 and Supplementary Tables 1-2
    main.tex         a small wrapper that compiles both on their own (written once, never overwritten)

The Claude Doc is where the text is edited (link in README.md). After an edit there, export it again
as Markdown over methods_draft.md and re-run:  python manuscript/methods/md_to_tex.py
Wording and numbers are copied as they are; only the markup changes.
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = {"Figure M1.": ("../figures/methods_overview.pdf", "0.6\\textwidth", "fig:methods-overview"),
           "Figure M2.": ("../figures/methods_example.pdf", "\\textwidth", "fig:methods-example")}
SPLIT_AT = "## Supplementary information"
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(s):
    """Plain text with Markdown emphasis -> LaTeX."""
    s = html.unescape(s).replace("\\[", "[").replace("\\]", "]")
    s = "".join(SPECIAL.get(c, c) for c in s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*(.+?)\*", r"\\textit{\1}", s)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    return s.replace("\u2013", "--").replace("\u2014", "---").rstrip()


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table(rows, caption):
    head, body = rows[0], rows[2:]
    n = len(head)
    first = ">{\\raggedright\\arraybackslash\\hspace{0pt}}X"
    rest = ">{\\raggedleft\\arraybackslash\\hspace{0pt}}X" if n > 4 else first      # number columns right-aligned
    spec = first + rest * (n - 1)
    if n == 4 and "What it records" in head:
        spec = "l>{\\raggedright\\arraybackslash}Xll"
    size = "\\scriptsize" if n > 6 else "\\footnotesize" if n > 4 else "\\small"
    out = ["\\begin{table}[htbp]", "\\centering", size, "\\setlength{\\tabcolsep}{3pt}" if n > 4 else ""]
    if caption:
        out.append("\\caption*{" + caption + "}")
    env = "tabularx" if "X" in spec else "tabular"
    out.append(f"\\begin{{{env}}}" + ("{\\textwidth}" if env == "tabularx" else "") + "{" + spec + "}")
    out += ["\\toprule", " & ".join("\\textbf{" + esc(c) + "}" for c in head) + " \\\\", "\\midrule"]
    out += [" & ".join(esc(c) for c in r) + " \\\\" for r in body]
    out += ["\\bottomrule", f"\\end{{{env}}}", "\\end{table}", ""]
    return out


def convert(lines, top):
    """top = Markdown heading level that becomes \\section."""
    names = ["section", "subsection", "subsubsection", "paragraph"]
    out, i, pending_caption, pending_figure = [], 0, None, False
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("#"):
            level, text = len(ln) - len(ln.lstrip("#")), ln.lstrip("#").strip()
            out += ["\\%s*{%s}" % (names[max(level - top, 0)], esc(text)), ""]
            i += 1
        elif ln.startswith("&#91;image:"):
            pending_figure = True
            i += 1
        elif ln.startswith("```"):
            lang, j = ln[3:].strip(), i + 1
            while not lines[j].startswith("```"):
                j += 1
            block = lines[i + 1:j]
            if lang == "latex":
                out += ["\\begin{equation*}"] + block + ["\\end{equation*}", ""]
            else:
                out += ["\\begin{lstlisting}"] + block + ["\\end{lstlisting}", ""]
            i = j + 1
        elif ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out += table([cells(r) for r in lines[i:j]], pending_caption)
            pending_caption, i = None, j
        elif re.match(r"(\d+\.|-) ", ln):
            env = "enumerate" if ln[0].isdigit() else "itemize"
            out.append(f"\\begin{{{env}}}")
            while i < len(lines) and re.match(r"(\d+\.|-) ", lines[i]):
                out.append("  \\item " + esc(re.sub(r"^(\d+\.|-) ", "", lines[i])))
                i += 1
            out += [f"\\end{{{env}}}", ""]
        else:
            text = esc(ln)
            label = re.match(r"\*\*((?:Supplementary )?(?:Table|Figure) \w+\.)\*\*", ln)
            if label and pending_figure:
                path, width, lab = FIGURES[label.group(1)]
                out += ["\\begin{figure}[htbp]", "\\centering", f"\\includegraphics[width={width}]{{{path}}}",
                        "\\caption*{" + text + "}", f"\\label{{{lab}}}", "\\end{figure}", ""]
                pending_figure = False
            elif label and "Table" in label.group(1):
                pending_caption = text
            else:
                out += [text, ""]
            i += 1
    return "\n".join(out).rstrip() + "\n"


MAIN = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, graphicx, booktabs, tabularx, array, caption, listings}
\lstset{basicstyle=\ttfamily\footnotesize, breaklines=true, breakatwhitespace=true, columns=fullflexible,
        keepspaces=true, frame=single, literate={°}{{\textdegree}}1}
\begin{document}
\section*{Methods}
\input{methods}
\clearpage
\section*{Supplementary information}
\input{supplement}
\end{document}
"""

if __name__ == "__main__":
    md = (HERE / "methods_draft.md").read_text(encoding="utf-8").split("\n")
    start = next(i for i, l in enumerate(md) if l.startswith("## "))      # skip title, date, author
    cut = md.index(SPLIT_AT)
    (HERE / "methods.tex").write_text(convert(md[start:cut], top=1), encoding="utf-8")
    (HERE / "supplement.tex").write_text(convert(md[cut + 1:], top=2), encoding="utf-8")
    if not (HERE / "main.tex").exists():
        (HERE / "main.tex").write_text(MAIN, encoding="utf-8")
    for f in ("methods.tex", "supplement.tex"):
        t = (HERE / f).read_text(encoding="utf-8")
        print(f, len(t.split("\n")), "lines;", t.count("\\begin{table}"), "tables,", t.count("\\begin{figure}"),
              "figures,", t.count("\\begin{equation*}"), "equations")
