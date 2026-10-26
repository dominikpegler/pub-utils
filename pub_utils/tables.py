# Utilities to generate consistent LaTeX tables from pandas DataFrames.
#
# Features
# - Single-/double-column floats: table vs table* via width="1col"/"2col".
# - threeparttable wrapper with notes and font size control.
# - Optional siunitx numeric alignment (S columns) with inferred widths.
# - Optional tabularx wrapping to fit \columnwidth or \textwidth.
# - Long tables via DataFrame.to_latex(longtable=True) (cannot be table*).
# - Configurable float placement (e.g., "!ht", "H", "tbp").
# - Header tweaks: center specific leaf headers (e.g., "\%") under S columns; optional header overrides.
# - Group cmidrules for MultiIndex top-level column groups; optional \midrule before specified rows.
# - Option to force specific numeric leaf headers to use 'r' instead of S.
# - NEW: group_cmidrules_skip to omit cmidrules for specified top-level group labels (e.g., empty stub).
#
# LaTeX preamble (recommended)
#   \usepackage{booktabs}
#   \usepackage{threeparttable}
#   \usepackage{siunitx}
#   \sisetup{detect-all}
#   % Optional for width control:
#   \usepackage{tabularx}
#   % If you use placement=H:
#   \usepackage{float}
#
# Notes
# - longtable cannot span two columns; do not use width="2col" for longtable.
# - When using siunitx S columns, escape=False is required.

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

import math
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


# @dataclass
# class TableSpec:
#     caption: str
#     label: str
#     # Layout
#     width: str = "1col"                  # "1col" or "2col"
#     star: Optional[bool] = None          # override; default derives from width
#     wrap: str = "threeparttable"         # "threeparttable" | "plain" | "longtable"
#     fontsize_pt: tuple[int, int] = (7, 9)
#     notes: Optional[str] = None
#     placement: Optional[str] = "!ht"     # e.g., "!ht", "H", "tbp"; None to omit []
#     # Formatting
#     use_siunitx: bool = True
#     si_table_format: Optional[str] = None  # e.g., "3.3"; overrides inference
#     column_format: Optional[str] = None
#     float_fmt: str = "{:.3f}"
#     multicolumn_format: str = "c"
#     add_cmidrule: bool = True
#     use_index: bool = True
#     top_addlinespace: str = r"\addlinespace[0.3em]"
#     # Width fitting
#     use_tabularx: bool = False
#     width_macro: str = "columnwidth"     # "columnwidth" or "textwidth"
#     # Header tweaks
#     header_overrides: Optional[dict[str, str]] = None   # map leaf header -> replacement text
#     center_leaf_headers: Optional[set[str]] = None      # leaf headers to wrap with \multicolumn{1}{c}{...}
#     # Column-format tweak: treat these numeric leaf headers as 'r' instead of S
#     numeric_as_r: Optional[set[str]] = None
#     # Group/row rule tweaks
#     group_cmidrules: bool = False                          # add cmidrules per top-level column group
#     midrule_before_rows: Optional[list[str]] = None        # insert \midrule before first row starting with any of these
#     # Hook for last-mile edits
#     post_latex: Optional[Callable[[str, "TableSpec"], str]] = None
#     align_text_as_r: Optional[set[str]] = None  # leaf headers to force right alignment even if text
#     # NEW: skip specified top-level labels when drawing group cmidrules
#     group_cmidrules_skip: Optional[set[str]] = None


# 23.10.2025: patched for 2-column functionality
@dataclass
class TableSpec:
    caption: str
    label: str
    # Layout
    width: str = "1col"                  # "1col" or "2col"
    star: Optional[bool] = None          # override; default derives from width
    wrap: str = "threeparttable"         # "threeparttable" | "plain" | "longtable"
    fontsize_pt: tuple[int, int] = (7, 9)
    # Interior column separation; None keeps the LaTeX default (6pt).
    # Smaller values (3-5pt) let wide tables fit the text width.
    tabcolsep_pt: Optional[int] = None
    notes: Optional[str] = None
    placement: Optional[str] = "!ht"     # e.g., "!ht", "H", "tbp"; None to omit []
    # Formatting
    use_siunitx: bool = True
    si_table_format: Optional[str] = None
    column_format: Optional[str] = None
    float_fmt: str = "{:.3f}"
    multicolumn_format: str = "c"
    add_cmidrule: bool = True
    use_index: bool = True
    top_addlinespace: str = r"\addlinespace[0.3em]"
    # Width fitting
    use_tabularx: bool = False
    width_macro: Optional[str] = None     # None = auto: 1col -> columnwidth, 2col -> textwidth
    # Header tweaks
    header_overrides: Optional[dict[str, str]] = None
    center_leaf_headers: Optional[set[str]] = None
    numeric_as_r: Optional[set[str]] = None
    group_cmidrules: bool = False
    midrule_before_rows: Optional[list[str]] = None
    post_latex: Optional[Callable[[str, "TableSpec"], str]] = None
    align_text_as_r: Optional[set[str]] = None
    group_cmidrules_skip: Optional[set[str]] = None
    # elsarticle variant: merge notes into the caption instead of tablenotes
    notes_in_caption: bool = False
    # original (Title Case) caption for the APA twin written to <stem>_apa.tex
    caption_apa: Optional[str] = None
    # column format for the APA twin (defaults to spec.column_format);
    # lets the elsarticle table wrap a column that the APA table does not
    column_format_apa: Optional[str] = None

    
    
def _decimals_from_floatfmt(fmt: str) -> int:
    m = re.search(r"\{:\.(\d+)f\}", fmt)
    return int(m.group(1)) if m else 3


def _infer_si_spec_for_series(s: pd.Series, decimals: int) -> tuple[int, int, int]:
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s.to_numpy(dtype="float64", na_value=np.nan))]
    if s.empty:
        return (1, decimals, 0)
    max_abs = float(np.nanmax(np.abs(s.to_numpy())))
    int_digits = 1 if max_abs < 1 else int(math.floor(math.log10(max_abs))) + 1
    sign_flag = int((s < 0).any())
    return (int_digits, decimals, sign_flag)


# def _build_column_format(df: pd.DataFrame, spec: TableSpec) -> str:
#     n_idx = df.index.nlevels
#     n_dat = df.shape[1]
#     if not spec.use_siunitx:
#         pieces = ["l" * n_idx]
#         for col in df.columns:
#             leaf = col[-1] if isinstance(col, tuple) else col
#             col_data = df[col]
#             if is_numeric_dtype(col_data):
#                 pieces.append("r")
#             else:
#                 if spec.align_text_as_r and str(leaf) in spec.align_text_as_r:
#                     pieces.append("r")
#                 else:
#                     pieces.append("l")
#         return "".join(pieces)

#     decimals = _decimals_from_floatfmt(spec.float_fmt)
#     pieces: list[str] = ["l" * n_idx]
#     for col in df.columns:
#         leaf = col[-1] if isinstance(col, tuple) else col
#         col_data = df[col]
#         if is_numeric_dtype(col_data):
#             if spec.numeric_as_r and str(leaf) in spec.numeric_as_r:
#                 pieces.append("r")
#             else:
#                 if spec.si_table_format:
#                     s_spec = f"S[table-format={spec.si_table_format}]"
#                 else:
#                     ints, decs, sign = _infer_si_spec_for_series(col_data, decimals)
#                     s_spec = (
#                         "S[table-figures-integer="
#                         f"{ints}"
#                         ",table-figures-decimal="
#                         f"{decs}"
#                         ",table-figures-sign="
#                         f"{sign}"
#                         "]"
#                     )
#                 pieces.append(s_spec)
#         else:
#             if spec.align_text_as_r and str(leaf) in spec.align_text_as_r:
#                 pieces.append("r")
#             else:
#                 pieces.append("l")
#     return "".join(pieces)

# 24.10.: upgrade to si v3

def _build_column_format(df: pd.DataFrame, spec: TableSpec) -> str:
    # when use_index is False the index column is not rendered by
    # df.to_latex, so the column format must not reserve index columns
    # (a spurious extra 'l' shifted every generated table's columns and
    # produced row-level overfull boxes; fixed 2026-09-23)
    n_idx = df.index.nlevels if spec.use_index else 0
    if not spec.use_siunitx:
        pieces = ["l" * n_idx]
        for col in df.columns:
            leaf = col[-1] if isinstance(col, tuple) else col
            col_data = df[col]
            if is_numeric_dtype(col_data):
                pieces.append("r")
            else:
                if spec.align_text_as_r and str(leaf) in spec.align_text_as_r:
                    pieces.append("r")
                else:
                    pieces.append("l")
        return "".join(pieces)

    decimals = _decimals_from_floatfmt(spec.float_fmt)
    # index columns are only reserved when the index is rendered
    # (see the use_index note above; fixed 2026-09-23)
    pieces: list[str] = ["l" * (df.index.nlevels if spec.use_index else 0)]
    for col in df.columns:
        leaf = col[-1] if isinstance(col, tuple) else col
        col_data = df[col]
        if is_numeric_dtype(col_data):
            if spec.numeric_as_r and str(leaf) in spec.numeric_as_r:
                pieces.append("r")
            else:
                if spec.si_table_format:
                    # explicit override, e.g., "2.3" or "+2.3"
                    s_spec = f"S[table-format={spec.si_table_format}]"
                else:
                    ints, decs, sign = _infer_si_spec_for_series(col_data, decimals)
                    # v2/v3-compatible: use table-format, include leading '+' if negatives present
                    sign_prefix = "+" if sign else ""
                    s_spec = f"S[table-format={sign_prefix}{ints}.{decs}]"
                pieces.append(s_spec)
        else:
            if spec.align_text_as_r and str(leaf) in spec.align_text_as_r:
                pieces.append("r")
            else:
                pieces.append("l")
    return "".join(pieces)



def _tabular_to_tabularx(latex_body: str, width_macro: str) -> str:
    latex_body = re.sub(
        r"\\begin{tabular}(\{.*?\})",
        rf"\\begin{{tabularx}}{{\\{width_macro}}}\1",
        latex_body,
        count=1,
    )
    latex_body = latex_body.replace(r"\end{tabular}", r"\end{tabularx}", 1)
    return latex_body


def _apply_header_overrides(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    if not (spec.header_overrides or spec.center_leaf_headers):
        return df
    # Build new columns
    new_cols = []
    had_tuple = False
    for col in df.columns:
        if isinstance(col, tuple):
            had_tuple = True
            head, leaf = col[:-1], col[-1]
        else:
            head, leaf = tuple(), col
        s = str(leaf)
        if spec.header_overrides and s in spec.header_overrides:
            s = spec.header_overrides[s]
        if spec.center_leaf_headers and s in spec.center_leaf_headers:
            s = rf"\multicolumn{{1}}{{c}}{{{s}}}"
        new_cols.append((*head, s) if head else s)
    out = df.copy()
    if had_tuple:
        out.columns = pd.MultiIndex.from_tuples(new_cols, names=df.columns.names)
    else:
        out.columns = new_cols
    return out


def _insert_group_cmidrules(body: str, df: pd.DataFrame, spec: TableSpec) -> str:
    if not spec.group_cmidrules:
        return body
    if not isinstance(df.columns, pd.MultiIndex):
        return body
    # Offset for index columns if the index is printed
    n_idx_offset = df.index.nlevels if spec.use_index else 0
    top = [c[0] for c in df.columns]
    spans = []
    i = 0
    while i < len(top):
        g = top[i]
        j = i
        while j < len(top) and top[j] == g:
            j += 1
        # columns are 1-based; data starts at n_idx_offset+1
        start = n_idx_offset + 1 + i
        end = n_idx_offset + j
        if spec.group_cmidrules_skip and (str(g) in spec.group_cmidrules_skip):
            i = j
            continue
        spans.append((start, end))
        i = j
    lines = body.splitlines()
    for k, ln in enumerate(lines):
        if r"\multicolumn" in ln and "&" in ln:
            insert_at = k + 1
            cmis = [rf"\cmidrule(lr){{{a}-{b}}}" for (a, b) in spans]
            lines[insert_at:insert_at] = cmis
            break
    return "\n".join(lines)


def _insert_midrule_before_rows(body: str, spec: TableSpec) -> str:
    """Insert \\midrule before every row whose first cell matches a listed pattern.

    Patterns are matched with startswith on the stripped row line. Callers
    passing a single pattern keep the former single-insert behavior because a
    matching row occurs once; callers passing several patterns get one midrule
    per matching row.
    """
    if not spec.midrule_before_rows:
        return body
    lines = body.splitlines()
    out = []
    for ln in lines:
        stripped = ln.strip()
        if any(stripped.startswith(pat) for pat in spec.midrule_before_rows):
            out.append(r"\midrule")
        out.append(ln)
    return "\n".join(out)


def _brace_s_column_headers(body: str, colfmt: str) -> str:
    """Wrap header cells in {…} for columns whose format starts with S."""
    col_types = []
    i = 0
    while i < len(colfmt):
        if colfmt[i] == 'S':
            j = i + 1
            if j < len(colfmt) and colfmt[j] == '[':
                depth = 1
                j += 1
                while j < len(colfmt) and depth > 0:
                    if colfmt[j] == '[': depth += 1
                    elif colfmt[j] == ']': depth -= 1
                    j += 1
            col_types.append('S')
            i = j
        elif colfmt[i] in 'lcrX':
            col_types.append(colfmt[i])
            i += 1
        else:
            i += 1

    lines = body.splitlines()
    in_header = True
    for k, ln in enumerate(lines):
        stripped = ln.strip()
        if (stripped.startswith(r'\toprule')
            or stripped.startswith(r'\addlinespace')
            or stripped.startswith(r'\begin')
            or stripped == ''):
            continue
        # Header rows end at the first \midrule; brace every header row so
        # that numeric leaf headers (e.g., "18") are typeset as text blocks
        # instead of overflowing the S-column width (fixed 2026-10-25).
        if stripped.startswith(r'\midrule'):
            break
        if '&' in stripped and stripped.endswith(r'\\'):
            # Strip trailing \\ before splitting
            content = ln.rstrip()
            if content.endswith(r'\\'):
                content = content[:-2]
            cells = content.split('&')
            n_cells = len(cells)
            # Align from the RIGHT: the last n_cells entries
            # in col_types correspond to the header cells
            if n_cells <= len(col_types):
                aligned_types = col_types[len(col_types) - n_cells:]
                new_cells = []
                for cell, ct in zip(cells, aligned_types):
                    cell_stripped = cell.strip()
                    # \multicolumn/\multirow must be the first token in a
                    # cell; wrapping them in braces raises "Misplaced \omit".
                    if (ct == 'S'
                            and 'multicolumn' not in cell_stripped
                            and 'multirow' not in cell_stripped
                            and not (cell_stripped.startswith('{') and cell_stripped.endswith('}'))):
                        new_cells.append(' {' + cell_stripped + '} ')
                    else:
                        new_cells.append(cell)
                lines[k] = ' & '.join(new_cells) + r' \\'
    return '\n'.join(lines)


def df_to_latex(df: pd.DataFrame, spec: TableSpec) -> str:
    # Header overrides/centering before rendering
    df = _apply_header_overrides(df, spec)

    n_idx = df.index.nlevels if spec.use_index else 0
    n_dat = df.shape[1]

    colfmt = spec.column_format or _build_column_format(df, spec)
    longtable = spec.wrap == "longtable"
    # For longtables the caption comes from df.to_latex; merge notes there
    # when they are meant to live in the caption.
    caption_for_latex = spec.caption
    if longtable and spec.notes_in_caption and spec.notes:
        caption_for_latex = f"{spec.caption} {spec.notes}"
    body = df.to_latex(
        index=spec.use_index,
        escape=False,
        multicolumn=True,
        multicolumn_format=spec.multicolumn_format,
        bold_rows=False,
        column_format=colfmt,
        multirow=False,
        float_format=spec.float_fmt.format,
        longtable=longtable,
        caption=(caption_for_latex if longtable else None),
        label=(spec.label if longtable else None),
    )

    # Space after \toprule
    body = body.replace(r"\toprule", r"\toprule" + "\n" + spec.top_addlinespace)

    if spec.use_siunitx:
        body = _brace_s_column_headers(body, colfmt)

    # Single-span cmidrule (legacy behavior)
    if spec.add_cmidrule and not longtable:
        start = n_idx + 1
        end = n_idx + n_dat
        lines = body.splitlines()
        for i, ln in enumerate(lines):
            if r"\multicolumn" in ln and "&" in ln:
                lines.insert(i + 1, rf"\cmidrule(lr){{{start}-{end}}}")
                break
        body = "\n".join(lines)

    # Per-group cmidrules
    body = _insert_group_cmidrules(body, df, spec)

    # Midrule before specified rows
    body = _insert_midrule_before_rows(body, spec)

    # Convert to tabularx if requested (not for longtable)
    
    # if spec.use_tabularx and not longtable:
    #     width_macro = spec.width_macro or ("textwidth" if spec.width == "2col" else "columnwidth")
    #     body = _tabular_to_tabularx(body, width_macro)

    # 23.10.2025: patched for 2-column functionality
    if spec.use_tabularx and not longtable:
        # auto width unless explicitly overridden
        width_macro = (
            spec.width_macro
            if spec.width_macro
            else ("textwidth" if spec.width == "2col" else "columnwidth")
        )
        body = _tabular_to_tabularx(body, width_macro)

        
    if spec.post_latex:
        body = spec.post_latex(body, spec)

    return body


# def wrap_table(latex_body: str, spec: TableSpec) -> str:
#     if spec.wrap == "longtable":
#         if (spec.star is True) or (spec.width == "2col"):
#             raise ValueError("longtable cannot span two columns; use width='1col' or restructure.")
#         return latex_body

#     star = spec.star if spec.star is not None else (spec.width == "2col")
#     tbl_env = "table*" if star else "table"
#     place_opt = f"[{spec.placement}]" if spec.placement else ""

    
# 23.10.2025: patched for better 2-column functionality

def wrap_table(latex_body: str, spec: TableSpec) -> str:
    if spec.wrap == "longtable":
        if (spec.star is True) or (spec.width == "2col"):
            raise ValueError("longtable cannot span two columns; use width='1col' or restructure.")
        if spec.fontsize_pt:
            latex_body = (
                f"\\begingroup\\fontsize{{{spec.fontsize_pt[0]}}}{{{spec.fontsize_pt[1]}}}\\selectfont\n"
                + (f"\\setlength{{\\tabcolsep}}{{{spec.tabcolsep_pt}pt}}\n" if spec.tabcolsep_pt else "")
                + f"{latex_body}\n"
                "\\endgroup"
            )
        elif spec.tabcolsep_pt:
            latex_body = (
                f"\\begingroup\\setlength{{\\tabcolsep}}{{{spec.tabcolsep_pt}pt}}\n"
                f"{latex_body}\n"
                "\\endgroup"
            )
        return latex_body


    star = spec.star if spec.star is not None else (spec.width == "2col")
    tbl_env = "table*" if star else "table"

    # Caption: merge notes into the caption in elsarticle mode;
    # otherwise keep the note out of the caption.
    caption = spec.caption
    if spec.notes_in_caption and spec.notes:
        caption = f"{spec.caption} {spec.notes}"

    # Safe placement for two-column floats under IEEEtran
    placement_effective = spec.placement
    if star:
        # Avoid H for star floats; IEEEtran won’t place them
        if placement_effective in ("H", None):
            placement_effective = "!t"
        elif placement_effective == "!ht":
            placement_effective = "!t"

    place_opt = f"[{placement_effective}]" if placement_effective else ""    
    # end patch
    
    parts: list[str] = []
    parts.append(fr"\begin{{{tbl_env}}}{place_opt}")
    parts.append(r"\centering")
    parts.append(fr"\caption{{{caption}}}")
    parts.append(fr"\label{{{spec.label}}}")
    if spec.wrap == "threeparttable":
        parts.append(r"\begin{threeparttable}")
    if spec.fontsize_pt:
        parts.append(fr"\fontsize{{{spec.fontsize_pt[0]}}}{{{spec.fontsize_pt[1]}}}\selectfont")
    if spec.tabcolsep_pt:
        parts.append(fr"\setlength{{\tabcolsep}}{{{spec.tabcolsep_pt}pt}}")
    parts.append(latex_body)
    if spec.wrap == "threeparttable":
        if spec.notes and not spec.notes_in_caption:
            parts.append(r"\begin{tablenotes}")
            parts.append(fr"\item \textit{{Note.}} {spec.notes}")
            parts.append(r"\end{tablenotes}")
        parts.append(r"\end{threeparttable}")
    parts.append(fr"\end{{{tbl_env}}}")
    return "\n".join(parts)


def _check_tex_ready(text: Optional[str], where: str) -> None:
    """
    Caption and notes are PRE-ESCAPED LaTeX text by contract. Fail loudly
    instead of silently emitting a broken table on the two known failure
    modes: a raw percent sign, and a bare underscore outside $...$ math spans
    (subscript in text mode -> "! Missing $ inserted").
    """
    if not text:
        return
    m = re.search(r"(?<!\\)%", text)
    if m:
        s = max(0, m.start() - 40)
        raise ValueError(
            f"write_table {where}: unescaped % at ...{text[s:m.start() + 41]!r}; "
            r"escape it as \% in the caption/notes before calling write_table."
        )
    stripped = re.sub(r"\$[^$]*\$", "", text)
    m = re.search(r"(?<!\\)_", stripped)
    if m:
        s = max(0, m.start() - 40)
        raise ValueError(
            f"write_table {where}: bare _ at ...{stripped[s:m.start() + 41]!r}; "
            r"avoid internal keys in prose or escape as \_."
        )


def write_table(df: pd.DataFrame, out_path: Path, spec: TableSpec) -> Path:
    _check_tex_ready(spec.caption, "caption")
    _check_tex_ready(spec.notes, "notes")
    _check_tex_ready(spec.caption_apa, "caption_apa")
    body = df_to_latex(df, spec)
    wrapped = wrap_table(body, spec)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(wrapped)

    # APA twin: the original style (notes in tablenotes, original caption) is
    # written under <stem>_apa.tex next to the elsarticle-style main file.
    # column_format_apa also switches the twin back to a plain tabular
    # (use_tabularx=False), because the twin replicates the original layout.
    if spec.notes_in_caption and spec.notes:
        apa_spec = replace(
            spec,
            notes_in_caption=False,
            caption=(spec.caption_apa if spec.caption_apa is not None else spec.caption),
            column_format=(spec.column_format_apa if spec.column_format_apa is not None else spec.column_format),
            use_tabularx=(False if spec.column_format_apa is not None else spec.use_tabularx),
        )
        apa_body = df_to_latex(df, apa_spec)
        apa_path = out_path.with_name(out_path.stem + "_apa" + out_path.suffix)
        apa_path.write_text(wrap_table(apa_body, apa_spec))
    return out_path


def write_styler(styler: pd.io.formats.style.Styler, out_path: Path, spec: TableSpec) -> Path:
    env = "longtable" if spec.wrap == "longtable" else "tabular"
    latex = styler.to_latex(
        environment=env,
        hrules=True,
        caption=(spec.caption if env == "longtable" else None),
        label=(spec.label if env == "longtable" else None),
    )
    
    # if spec.wrap != "longtable":
    #     latex = latex.replace(r"\toprule", r"\toprule" + "\n" + spec.top_addlinespace)
    #     if spec.use_tabularx:
    #         width_macro = spec.width_macro or ("textwidth" if spec.width == "2col" else "columnwidth")
    #         latex = _tabular_to_tabularx(latex, width_macro)
    #     latex = wrap_table(latex, spec)
    
    # 23.10.2025: patched for better 2-column functionality
    if spec.wrap != "longtable":
        latex = latex.replace(r"\toprule", r"\toprule" + "\n" + spec.top_addlinespace)
        if spec.use_tabularx:
            width_macro = (
                spec.width_macro
                if spec.width_macro
                else ("textwidth" if spec.width == "2col" else "columnwidth")
            )
            latex = _tabular_to_tabularx(latex, width_macro)
        latex = wrap_table(latex, spec)

        
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    return out_path

    
def fmt_p(p):
    """
    Formats a p-value to a specified number of digits for tables.
    """
    try:
        return "<.001" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return str(p)
