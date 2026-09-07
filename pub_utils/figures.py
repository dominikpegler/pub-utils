from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots  # registers SciencePlots styles
import numpy as np
import fitz  # pip install PyMuPDF
from PIL import Image

MM_PER_IN = 25.4
def mm_to_in(mm): return mm / MM_PER_IN

def setup_style(
    profile="nature",              # "ieee" (serif) or "nature" (sans)
    use_tex=True,
    base_font=7,
    title_font_delta=2,
    minor_ticks=True,
    major_ticks=True,
    dpi_out=300,
    dpi_out_savefig=300,
    latex_engine="pdflatex",     # "pdflatex" or "xelatex"/"lualatex"
    reset=True,
    rc_extra=None,               # optional dict of user overrides

):
    if reset:
        plt.rcdefaults()
        plt.style.use("default")

    if profile == "ieee":
        plt.style.use(["science", "ieee"])
    else:
        plt.style.use(["science", "nature"])
        
    rc = {
        "font.size": base_font,
        "axes.titlesize": base_font + title_font_delta,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font,
        "ytick.labelsize": base_font,
        "legend.fontsize": base_font,
        "figure.dpi": dpi_out,
        "savefig.dpi": dpi_out_savefig,

        # sides and minor ticks (SciencePlots leaves this reasonable; we expose as knobs)
        "xtick.bottom": major_ticks,
        "xtick.top": major_ticks,
        "ytick.left": major_ticks,
        "ytick.right": major_ticks,
        "xtick.minor.visible": minor_ticks,
        "ytick.minor.visible": minor_ticks,

        # crucial delta so hist bars don’t hide ticks
        "axes.axisbelow": False,
        
        # Ensure TrueType embedding in PDF/PS; keep SVG text as text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "svg.fonttype": "none",
    }


    
    if use_tex:
        rc["text.usetex"] = True
        rc["pgf.texsystem"] = latex_engine
        if profile == "ieee":
            rc["text.latex.preamble"] = r"\usepackage[T1]{fontenc}\usepackage{siunitx}\usepackage{newtxtext,newtxmath}"
            rc["font.family"] = "serif"
            rc["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "Times"]
        else:
            rc["text.latex.preamble"] = r"\usepackage[T1]{fontenc}\usepackage{siunitx}\usepackage{helvet}\usepackage{sfmath}\renewcommand{\familydefault}{\sfdefault}"
            rc["font.family"] = "sans-serif"
            rc["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "Nimbus Sans"]
    else:
        rc["text.usetex"] = False
        if profile == "ieee":
            rc["font.family"] = "serif"
            rc["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif", "Times"]
            rc["mathtext.fontset"] = "stix"
        else:
            rc["font.family"] = "sans-serif"
            rc["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial", "Nimbus Sans"]
            rc["mathtext.fontset"] = "dejavusans"
    if rc_extra:
        rc.update(rc_extra)

    mpl.rcParams.update(rc)


def make_fig(
    width_mm=89,
    height_mm=45,
    nrows=1,
    ncols=1,
    sharex=False,
    sharey=False,
    box_aspect=None,
    constrained=True,     # new (default keeps prior behavior True)
    flatten=True,         # new (default keeps returning flattened list like before)
    facecolor=None,       # new (optional)
    **kwargs
):
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(mm_to_in(width_mm), mm_to_in(height_mm)),
        constrained_layout=constrained, sharex=sharex, sharey=sharey,
        facecolor=facecolor,
        **kwargs
    )
    if hasattr(axs, "ravel") and flatten:
        axs = axs.ravel()
    elif not hasattr(axs, "ravel"):
        axs = [axs]
    if box_aspect is not None:
        # Apply to each axes (works in both 1D list and 2D array cases)
        if isinstance(axs, (list, tuple)):
            targets = axs
        else:
            # axs is 2D ndarray
            targets = axs.flat
        for ax in targets:
            ax.set_box_aspect(box_aspect)
    return fig, axs


def make_grid(
    width_mm=89,
    panels=(1, 1),            # (rows, cols)
    panel_aspect=1.0,         # height/width per panel (data box aspect target)
    sharex=False,
    sharey=False,
    margins=(0.07, 0.05, 0.995, 0.96),  # left, bottom, right, top
    gutter=(0.12, 0.12),      # wspace, hspace (relative to average axes size)
    facecolor="#fff",
    constrained=False,        # match your custom subplots_adjust workflow
    flatten=False,            # return 2D axes array for row/col indexing
):
    rows, cols = panels
    # Overall height scaled by panel_aspect and grid shape
    height_mm = width_mm * (rows / max(cols, 1)) * panel_aspect
    fig, axs = make_fig(
        width_mm=width_mm, height_mm=height_mm,
        nrows=rows, ncols=cols,
        sharex=sharex, sharey=sharey,
        box_aspect=None,
        constrained=constrained,
        flatten=flatten,
        facecolor=facecolor,
    )
    if not constrained:
        l, b, r, t = margins
        w, h = gutter
        fig.subplots_adjust(left=l, bottom=b, right=r, top=t, wspace=w, hspace=h)
    return fig, axs


def label_panels(axs, start="a", x=0.0, y=1.05, weight="bold", fontsize=9):
    import string
    def _iter_axes(axs_obj):
        if isinstance(axs_obj, (list, tuple)):
            for a in axs_obj: yield a
        elif hasattr(axs_obj, "flat"):
            for a in axs_obj.flat: yield a
        else:
            yield axs_obj
    for i, ax in enumerate(_iter_axes(axs)):
        ax.text(x, y, string.ascii_lowercase[i + (ord(start)-97)],
                transform=ax.transAxes, fontweight=weight, fontsize=fontsize)


def super_labels(fig, xlabel=None, ylabel=None, xoffset=0.0, yoffset=0.0, rotation=90):
    if ylabel is not None:
        fig.text(0.0 - xoffset, 0.5 - yoffset, ylabel, rotation=rotation, va="center")
    if xlabel is not None:
        fig.text(0.5 - xoffset, 0.0 - yoffset, xlabel, va="top")


def save(fig, out_base: Path, formats=("pdf", "png"), dpi_png=600, tight=False):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        if ext == "png":
            fig.savefig(out_base.with_suffix(".png"),
                        dpi=dpi_png,
                        bbox_inches=("tight" if tight else None))
        elif ext == "pgf":
            try:
                fig.savefig(out_base.with_suffix(".pgf"),
                            bbox_inches=("tight" if tight else None))
            except Exception:
                pass
        else:
            fig.savefig(out_base.with_suffix(f".{ext}"),
                        bbox_inches=("tight" if tight else None))

            
def bins_fd(x, max_bins=None):
    """
    Freedman–Diaconis bin estimator. Returns an integer number of bins.
    Use only if you want an automatic rule; otherwise pass bins explicitly.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return 1
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr == 0:
        return min(max_bins or 50, max(1, int(np.sqrt(x.size))))
    h = 2 * iqr * (x.size ** (-1 / 3))
    if h <= 0:
        return min(max_bins or 50, max(1, int(np.sqrt(x.size))))
    nbins = int(np.ceil((x.max() - x.min()) / h)) or 1
    return min(nbins, max_bins) if max_bins else nbins


def set_tick_sides(ax, *, bottom=True, left=True, top=False, right=False):
    ax.tick_params(axis="x", which="both", bottom=bottom, top=top)
    ax.tick_params(axis="y", which="both", left=left, right=right)

    
def hist_panel(
    ax,
    x,
    *,
    bins=None,
    range=None,
    density=False,
    color="steelblue",
    edgecolor="white",
    linewidth=0.5,
    alpha=1.0,
    grid=True,
    grid_axis="y",
    mean_line=True,
    mean_color="red",
    mean_ls="-",
    mean_lw=1.0,
    sd_lines=True,
    sd_color="black",
    sd_ls="--",
    sd_lw=1.0,
    sd_k=1.0,
    stats=True,
    stats_mode="box",
    stats_loc="upper right",
    stats_fmt="$N={N:d}$, $M={mean:.2f}$, $SD={sd:.2f}$",
    ddof=1,
    title_prefix=None,
    ci_quantiles=None,
    draw_ci=False,
    ci_color="red",
    ci_ls="--",
    ci_lw=1.0,
    stats_box_kw=None,   # e.g., dict(alpha=0.5, facecolor="white", edgecolor="none")
    weights=None # e.g., for relative frequncy / percent 
):

    arr = np.asarray(x)
    arr = arr[np.isfinite(arr)]
    N = int(arr.size)
    mean = float(np.mean(arr)) if N else np.nan
    sd = float(np.std(arr, ddof=ddof)) if N else np.nan

    n, edges, patches = ax.hist(
        arr, bins=bins, range=range, density=density,
        color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, weights=weights
    )

    if grid:
        ax.grid(True, axis=grid_axis, which="major")

    artists = []
    if N and np.isfinite(mean):
        if mean_line:
            artists.append(ax.axvline(mean, color=mean_color, linestyle=mean_ls, linewidth=mean_lw, label="Mean"))
        if sd_lines and np.isfinite(sd) and sd > 0:
            artists.append(ax.axvline(mean + sd_k*sd, color=sd_color, linestyle=sd_ls, linewidth=sd_lw, label=f"+{sd_k} SD"))
            artists.append(ax.axvline(mean - sd_k*sd, color=sd_color, linestyle=sd_ls, linewidth=sd_lw, label=f"-{sd_k} SD"))

    ci_lo = np.nan
    ci_hi = np.nan
    ci_artists = []
    if ci_quantiles is not None and N:
        try:
            q_lo, q_hi = ci_quantiles
            ci_lo, ci_hi = np.percentile(arr, [q_lo, q_hi])
            if draw_ci and np.isfinite([ci_lo, ci_hi]).all():
                ci_artists.append(ax.axvline(ci_lo, color=ci_color, linestyle=ci_ls, linewidth=ci_lw))
                ci_artists.append(ax.axvline(ci_hi, color=ci_color, linestyle=ci_ls, linewidth=ci_lw))
        except Exception:
            pass

    stats_artist = None
    stats_text = stats_fmt.format(N=N, mean=mean, sd=sd, ci_lo=ci_lo, ci_hi=ci_hi)
    if stats and stats_mode == "box":
        ha = "right" if "right" in stats_loc else "left"
        va = "top" if "upper" in stats_loc else "bottom"
        bbox_default = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none")
        if stats_box_kw:
            bbox_default.update(stats_box_kw)
        stats_artist = ax.text(
            0.98 if "right" in stats_loc else 0.02,
            0.98 if "upper" in stats_loc else 0.02,
            stats_text,
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=mpl.rcParams.get("font.size", 7),
            bbox=bbox_default,
        )
    elif stats and stats_mode == "title":
        base = ax.get_title()
        prefix = (title_prefix + "\n") if title_prefix else ""
        ax.set_title(prefix + (base + ("\n" if base else "")) + stats_text)

    return {
        "N": N,
        "mean": mean,
        "sd": sd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "counts": n,
        "edges": edges,
        "artists": artists,
        "ci_artists": ci_artists,
        "stats_artist": stats_artist, 
        "patches": patches, 
    }


           
def hide_interior_labels(axs, keep_bottom=True, keep_left=True, apply_x=True, apply_y=True):
    """
    Hide ticklabels on interior subplots of a 2D axes array.
    Keeps bottom row x labels and left column y labels (configurable).
    """
    if not hasattr(axs, "shape") or axs.ndim != 2:
        return
    rows, cols = axs.shape
    for r in range(rows):
        for c in range(cols):
            ax = axs[r][c]
            if apply_x:
                show_x = (r == rows - 1) if keep_bottom else False
                ax.tick_params(labelbottom=show_x)
            if apply_y:
                show_y = (c == 0) if keep_left else False
                ax.tick_params(labelleft=show_y)

                
                
def image_grid(
    image_paths,
    rows: int,
    cols: int,
    *,
    width_mm: float = 89,
    panel_aspect: float = 1.0,
    facecolor=None,
    constrained: bool = True,
    labels=None,         # optional list like ["a", "b", ...]; if None, no panel labels
    label_kw=None,       # e.g., dict(x=0.0, y=1.05, weight="bold")
):
    """
    Render a grid of images (paths or numpy arrays) using make_grid.
    Hides axes/ticks, preserves consistent sizing, and optionally places panel labels.

    image_paths: sequence of Path/str (read via imread) or numpy arrays.
    """
    import matplotlib.image as mpimg

    fig, axs = make_grid(
        width_mm=width_mm,
        panels=(rows, cols),
        panel_aspect=panel_aspect,
        constrained=constrained,
        flatten=False,
        facecolor=facecolor,
    )

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axs[r][c]
            # Clean up axes for image display
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.set_frame_on(False)

            if idx < len(image_paths):
                p = image_paths[idx]
                img = mpimg.imread(p) if isinstance(p, (str, Path)) else p
                ax.imshow(img)
            idx += 1

    if labels:
        kw = dict(x=0.0, y=1.05, weight="bold")
        if label_kw:
            kw.update(label_kw)
        for i in range(min(len(labels), rows * cols)):
            r, c = divmod(i, cols)
            axs[r][c].text(
                kw["x"], kw["y"], labels[i],
                transform=axs[r][c].transAxes,
                fontweight=kw["weight"]
            )
    return fig, axs


def bar_panel(
    ax,
    x,
    heights,
    *,
    labels=None,
    colors=None,
    edgecolor="black",
    linewidth=0.3,
    grid=True,
    grid_axis="y",
    ylabel=None,
    xlabel=None,
    rotate_xt=0,
    xt_ha="center",
    annotate=None,   # dict(text=..., loc="upper right", fontsize=9, box_kw={...})
):
    """
    Generic categorical bar chart helper with consistent styling and an optional
    annotation box (e.g., sample size, CI summaries). Complements hist_panel
    (which targets continuous histograms).
    """
    ax.bar(x, heights, color=colors, edgecolor=edgecolor, linewidth=linewidth)
    if grid:
        ax.grid(True, axis=grid_axis, which="major", alpha=0.3)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if labels is not None:
        ax.set_xticks(x, labels=labels, rotation=rotate_xt, ha=xt_ha)
    if annotate and annotate.get("text"):
        loc = annotate.get("loc", "upper right")
        ha = "right" if "right" in loc else "left"
        va = "top" if "upper" in loc else "bottom"
        box_kw = dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3")
        box_kw.update(annotate.get("box_kw", {}))
        ax.text(
            0.98 if "right" in loc else 0.02,
            0.97 if "upper" in loc else 0.03,
            annotate["text"],
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=annotate.get("fontsize", 9),
            bbox=box_kw,
        )
    return ax


def file_dimensions(directory, basename, print_only=False):
    pdf_path = Path(directory, f"{basename}.pdf")
    png_path = Path(directory, f"{basename}.png")

    dims = {}

    d = fitz.open(pdf_path)
    p = d[0]
    r = p.rect
    w_p = r.width
    h_p = r.height
    m_p = 25.4 / 72.0
    w_m_p = round(w_p * m_p, 2)
    h_m_p = round(h_p * m_p, 2)
    dims["PDF"] = {}
    dims["PDF"]["W"] = w_m_p
    dims["PDF"]["H"] = h_m_p

    i = Image.open(png_path)
    w_x = i.width
    h_x = i.height
    d_i = i.info.get("dpi", (72, 72))
    x_d = d_i[0]
    y_d = d_i[1]
    w_m_i = round(w_x / x_d * 25.4, 2)
    h_m_i = round(h_x / y_d * 25.4, 2)
    dims["PNG"] = {}
    dims["PNG"]["W"] = w_m_i
    dims["PNG"]["H"] = h_m_i
    dims["PNG"]["DPI_X"] = x_d
    dims["PNG"]["DPI_Y"] = y_d
    if print_only:
        print(f"PDF: W: {w_m_p:.1f}mm x H:{h_m_p:.1f}mm")
        print(f"PNG: W: {w_m_i:.1f}mm x H:{h_m_i:.1f}mm ({x_d:.0f}x{y_d:.0f} dpi)")
    else:
        return dims


def recolor_hist_by_split(patches, edges, split=0.0, below_color="red", above_color="green"):
    try:
        # color by bin midpoints relative to split
        for i in range(min(len(patches), len(edges) - 1)):
            mid = 0.5 * (edges[i] + edges[i + 1])
            patches[i].set_facecolor(below_color if mid < split else above_color)
    except Exception:
        pass
    return patches

