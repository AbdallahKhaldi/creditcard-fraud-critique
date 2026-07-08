# Credit-Card Fraud Detection — A Critical Reproduction

Final project for the University of Haifa course *שימוש בשיטות של דאטה סיינס בסייבר*
(Data Science in Cyber, 203.3888).

## What this project does

I took a public credit-card fraud detection tutorial that reports near-perfect
results and tried to reproduce it. The reported numbers turn out to be inflated by a
**data-leakage bug**: the tutorial applies SMOTE oversampling to the whole dataset
*before* splitting into train and test, so synthetic fraud examples leak into the
test set. On top of that it leans on accuracy and ROC-AUC, which are misleading when
only 0.17% of transactions are fraud.

The notebook reproduces the flawed result, shows mechanically why it is wrong, and
then redoes the analysis with a leakage-free protocol (resampling inside an imblearn
pipeline, on training folds only) and imbalance-aware metrics (precision/recall,
F-beta, MCC, PR-AUC).

## Main finding

The same logistic regression scores **F1 0.979** under the leaky protocol and
**F1 0.097** when evaluated honestly; nothing changed except where SMOTE runs. A
properly evaluated random forest reaches **F1 0.830 / MCC 0.836 / PR-AUC 0.807**,
catching 105 of 142 test frauds with 6 false alarms in 85,118 transactions. That is
a useful detector, and nowhere near the "perfect" numbers the tutorial advertises.
The tutorial's central claim is not supported by its evidence. Full details in the
notebook and in `report/report.pdf`.

## Links

- **Tutorial / source critiqued:** https://github.com/sergi-s/Credit-Card-fraud-detection
- **Original code repository:** https://github.com/sergi-s/Credit-Card-fraud-detection (same URL — the tutorial *is* the notebook-based repo)
- **Dataset:** ULB "Credit Card Fraud Detection" — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Dataset

284,807 transactions from European cardholders over two days in September 2013, 492
of them fraudulent (0.173%). Features are `Time`, `Amount`, `Class` (1 = fraud), and
`V1`–`V28`, which are anonymized PCA components. The CSV is ~144 MB and is **not**
committed to this repo.

Download `creditcard.csv` from the Kaggle link above and place it in `data/`
(instructions in `data/README.md`).

## How to run

Requires **Python 3.12+** (developed on 3.14).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# download creditcard.csv from Kaggle into data/  (see data/README.md)

jupyter notebook notebook/fraud_critique.ipynb
```

Run the notebook top-to-bottom on a fresh kernel ("Restart & Run All").

### Notes for a fresh run

- A free Kaggle account is needed for the dataset download.
- The full run takes a few minutes (about 4-5; the cross-validation and scaler
  ablation re-fit the models several times) and stays under ~2 GB of RAM; everything is
  seeded with `random_state=42`, so re-runs reproduce the same numbers.
- Start Jupyter from the repo root or from `notebook/`; the first cell resolves
  paths by looking for `requirements.txt` and stops with a clear message if the CSV
  or the repo can't be found.
- The flawed-reproduction section (3) is *supposed* to print near-perfect scores;
  that is the exhibit being debunked, not a result of this project.

## Repo layout

```
notebook/fraud_critique.ipynb   # the full analysis (sections 1-7)
src/utils.py                    # small plotting / metric helpers
report/report.md, report.pdf    # the written report (8 sections)
data/                           # put creditcard.csv here (gitignored)
```
