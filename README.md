# Learning to Rank on the Cranfield Collection

**Group:** info fetch
**Course:** Information Retrieval (CS60092), Autumn 2026-27, Assignment 3
**Repository:** <https://github.com/sumitmukherjee7/cranfield-learning-to-rank>

| Member | Roll number |
|---|---|
| Sumit Mukherjee | 23EC3AI17 |
| Abyan Hussain | 23EC3AI19 |
| Kinnar Halder | 23IE35001 |
| Avik Ghosh | 23EC10099 |

**Language:** Python 3, the preferred language for the assignment. No other language is used.

**Report:** [`report.pdf`](report.pdf)

**Ranked documents for the test queries:** [`results/methodA_test_ranking.txt`](results/methodA_test_ranking.txt) (Method A) and [`results/methodB_test_ranking.txt`](results/methodB_test_ranking.txt) (Method B). These are already included, so nothing needs to be run to see them.

---

## 1. What the code does

It implements two learning-to-rank methods on the Cranfield collection (1400 documents, 225 queries). Each method was implemented as the assignment specifies and then improved:

| System | Test MAP | Test F1@10 |
|---|---|---|
| Method A: Boolean weighted zone scoring, zone weights learned by grid search | 0.2694 | 0.2271 |
| Method A+: same, with IDF-weighted Boolean term matches (**final Method A**) | 0.3048 | 0.2588 |
| Method B: linear SVM over the 22 specified features | 0.3248 | 0.2639 |
| Method B+: adds an IDF Boolean feature, tuned BM25 and a pseudo-relevance-feedback feature (**final Method B**) | 0.3685 | 0.2965 |
| BM25 baseline (k1 = 1.2, b = 0.75) | 0.3187 | 0.2609 |

The 225 queries are split into 157 training and 68 test queries (fixed random seed 42). All weights and parameters are learned on the training queries only. The test queries are used only for the final evaluation.

---

## 2. Folder contents

```
info_fetch_A3/
|-- README.md                  this file
|-- requirements.txt           Python dependencies
|-- report.pdf                 the assignment report
|-- cran/                      Cranfield data (from cran.tar.gz)
|   |-- cran.all.1400          the 1400 documents
|   |-- cran.qry               the 225 queries
|   |-- cranqrel               relevance judgements
|   `-- cranqrel.readme
|-- src/                       source code
|   |-- run_experiments.py     main program: runs all experiments
|   |-- cranfield.py           parsing, preprocessing, inverted index, evaluation measures
|   |-- method_a_zone.py       Method A: Boolean zone scores and the weight grid search
|   |-- features.py            Method B: the 22 features, plus the PRF and zone-BM25 features
|   `-- method_b_svm.py        Method B: pointwise and pairwise (RankSVM) linear SVM
`-- results/                   ranked documents for the 68 test queries
    |-- methodA_test_ranking.txt         final Method A (A+)
    |-- methodB_test_ranking.txt         final Method B (B+)
    |-- methodA_basic_test_ranking.txt   Method A exactly as specified
    `-- methodB_basic_test_ranking.txt   Method B exactly as specified
```

**Ranking file format** (TREC run format, top 100 documents per test query, one line per document):

```
query_id Q0 doc_id rank score run_tag
1 Q0 51 1 3.535814 MethodB_plus_LinearSVM_pairwise
```

`query_id` is the query number from 1 to 225 as used in `cranqrel`. The `.I` numbers inside `cran.qry` skip values (001, 002, 004, 008, ...), so the code numbers the queries by their position in the file, which matches `cranqrel`.

---

## 3. Setting up the environment

The steps below were tested on Windows 11 with Python 3.12 and Python 3.14. The code uses only standard Python 3.10 syntax, so any Python 3.10 or newer should work.

### Step 1: Install Python

Check whether Python 3.10 or newer is installed:

```bash
python --version
```

On Linux and macOS, use `python3 --version`. On Windows, `py --version` also works.

If Python is missing, install it from <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"** in the installer.

### Step 2: Unzip and open a terminal in the folder

Extract `info_fetch_A3.zip`. This creates the folder `info_fetch_A3/`. Open a terminal (PowerShell or Command Prompt on Windows, Terminal on Linux and macOS) and change into it:

```bash
cd path/to/info_fetch_A3
```

All commands below are run from this folder. If you cloned the GitHub repository instead, the folder is called `cranfield-learning-to-rank` and has the same contents.

### Step 3: Create and activate a virtual environment

This keeps the dependencies separate from the rest of your system.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use Command Prompt instead.

**Windows (Command Prompt):**
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux and macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation the prompt starts with `(.venv)`, and `python` refers to the environment's Python.

### Step 4: Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs numpy, scipy, scikit-learn and nltk. NLTK is used only for its Porter stemmer, which needs no extra data download.

---

## 4. Running the code

The data is already in `cran/`. With the virtual environment active, run:

```bash
python src/run_experiments.py
```

This takes about 15 to 40 minutes, depending on the machine. Progress is printed with timestamps. The program:

1. parses and indexes the collection and splits the queries into 157 training and 68 test queries;
2. **Method A:** learns the zone weights by grid search for each Boolean match definition and objective;
3. **Method B:** computes the 22 features, then chooses the SVM formulation (pointwise or pairwise) and C by 5-fold cross-validation on the training queries, and runs the ablations;
4. **Improvements:** tunes BM25 and pseudo-relevance feedback on the training queries, evaluates each improvement step, and selects Method B+ by cross-validation;
5. evaluates all systems on the test queries (precision, recall and F1 at 5, 10 and 20, MAP, nDCG, significance tests) and writes the ranking files;
6. runs a 5-fold cross-validation over all 225 queries.

The last line printed is `wrote results/results.json`.

### Output

All output goes to `results/`:

| File | Contents |
|---|---|
| `methodA_test_ranking.txt`, `methodB_test_ranking.txt` | final rankings for the test queries (overwrites the included files) |
| `methodA_basic_test_ranking.txt`, `methodB_basic_test_ranking.txt` | rankings of the methods as specified (overwrites the included files) |
| `baseline_bm25_test_ranking.txt`, `baseline_bm25_prf_test_ranking.txt`, `baseline_vsm_test_ranking.txt` | baseline rankings |
| `results.json` | every number in the report's tables |
| `per_query_test.csv` | per-query P@10, R@10, F1@10 and AP |

All random seeds are fixed, so the ranking files are overwritten with identical results. We checked this: a clean install of this code on Python 3.12 reproduced all four included ranking files byte for byte.

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or another package) | The virtual environment is not active, or step 4 was skipped. Activate it (step 3) and run `python -m pip install -r requirements.txt`. |
| `FileNotFoundError: ... cran.all.1400` | The data is not in `cran/`. Extract the course's `cran.tar.gz` so that `cran/cran.all.1400`, `cran/cran.qry` and `cran/cranqrel` exist. |
| `python` not found on Windows | Use `py` instead of `python`, or reinstall Python with "Add python.exe to PATH" ticked. |
| Numbers differ slightly from the report | Possible with much older or newer library versions. The report was produced with the versions noted in `requirements.txt`. |
