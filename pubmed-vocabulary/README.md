# Vocabulary Change in PubMed Abstracts from 2015 to 2026

Code and data for:

De Winter, J. C. F. (2026). *“Crucial research findings remain”: Vocabulary change in PubMed abstracts from 2015 to 2026*. https://joostdewinter.github.io/pubmed-vocabulary.pdf

## Contents

| File | Content |
|---|---|
| `reproduce.py` | Runs the full analysis: screening, grouping, 30-word selection, cross-validation, lower bound and figures |
| `engine.py`, `lower_bound_analysis.py`, `pooled_outputs.py` | Functions used by `reproduce.py` |
| `check_lower_bound_gradient.py` | Numerical check of the confidence-interval calculation |
| `inputs/monthly_length_word_counts.npz` | Monthly counts of abstracts containing each of 22,580 words, by PMID half and 50-word length band |
| `candidates.json`, `Word_inclusion.csv` | Eligibility list (4,742 words) and all word-level decisions |
| `Word_audit_labels.csv`, `Word_audit_rubric.md` | Audit of the eligibility list |
| `Exclusions.csv`, `Eligibility_summary.json` | Excluded records |
| `Summary.csv` and the other `.csv`, `.json` and `.npz` files | Results reported in the paper |
| `Figure1_coverage`, `Figure2_projected_coverage`, `Figure3_lower_bound` | Figures 1–3 (PNG and SVG) |

## Running the analysis

Requirements: Python 3 with the packages in `requirements.txt` (`pip install -r requirements.txt`).

`python reproduce.py` runs the full analysis. It needs the per-abstract word indicators in a folder `features/` (three archives, about 1 GB in total), which are not included here because of their size. The full run needs about 20 GB of memory and 15 GB of free disk space.

The lower bounds and Figure 2 can be recomputed from the files in this folder with `python -c "import pooled_outputs as po; po.plot_projection(po.ai_bounds())"`.

## Data

PubMed data: Courtesy of the U.S. National Library of Medicine. The files reflect PubMed as of 19 September 2026 and do not reflect the most current data available from NLM. They contain PMIDs, journal identifiers, counts and word indicators, and no abstract text.

## License

The code and results are released under the MIT License (see `LICENSE`). Please cite the paper when you use them.
