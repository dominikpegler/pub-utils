# pub-utils

Shared publication-ready figure and table helpers, used as a git submodule by
the EchoChains, trust-bandits, joptim, spidernets, and bot-bandits analysis
repos. This is the single source of truth for these helpers; do not copy the
files into other repos.

## Setup

    conda activate echochain
    pip install -e .          # from this repo, in the echochain conda env

Consuming repos add this repo as a git submodule (e.g. at `pub-utils/`) and
install it editable:

    git submodule add git@github.com:dominikpegler/pub-utils.git pub-utils
    pip install -e ./pub-utils

## Contents

- `pub_utils/figures.py` — matplotlib style setup, figure/panel helpers,
  histograms, image grids, bar panels, and save utilities.
- `pub_utils/tables.py` — LaTeX table generation (`TableSpec`, `df_to_latex`,
  `wrap_table`, `write_table`, `write_styler`).

## Dependencies

- `figures.py`: matplotlib, scienceplots, numpy, PyMuPDF (fitz), Pillow
- `tables.py`: pandas, numpy
