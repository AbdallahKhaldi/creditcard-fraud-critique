# Data folder

The dataset is **not** stored in this repo (it's ~144 MB and gitignored).

Download `creditcard.csv` from the ULB "Credit Card Fraud Detection" dataset on Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file here so the path is:

```
data/creditcard.csv
```

A free Kaggle account is required. If you use the Kaggle CLI (not included in
`requirements.txt`; needs `pip install kaggle` and an API token in
`~/.kaggle/kaggle.json`):

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```
