"""Shared helpers for the fraud-critique notebook. Kept deliberately thin."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


def fraud_rate_table(bands, y):
    """Rows, fraud count and fraud rate per band.

    Reports rates rather than counts, because under 0.17% prevalence a band's
    raw fraud count mostly tracks how busy the band is.
    """
    # reindex so a band with zero fraud (or zero legit) rows still yields both columns
    table = pd.crosstab(bands, y).reindex(columns=[0, 1], fill_value=0)
    table.columns = ["legit", "fraud"]
    table["total"] = table["legit"] + table["fraud"]
    table["fraud_rate_%"] = 100 * table["fraud"] / table["total"]
    return table


def plot_fraud_rate(table, title, ax=None, color="tab:red"):
    """Bar chart of the fraud_rate_% column of a fraud_rate_table result."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    ax.bar(table.index.astype(str), table["fraud_rate_%"], color=color)
    ax.set_ylabel("fraud rate (%)")
    ax.set_title(title)
    return ax


def metrics_row(y_true, y_pred, y_score):
    """One row of headline metrics for the model-comparison tables.

    y_score is the positive-class probability (or decision score), which the
    two AUC metrics need; everything else works off the hard predictions.
    """
    return {
        "precision": metrics.precision_score(y_true, y_pred),
        "recall": metrics.recall_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
        "f2": metrics.fbeta_score(y_true, y_pred, beta=2),
        "mcc": metrics.matthews_corrcoef(y_true, y_pred),
        "roc_auc": metrics.roc_auc_score(y_true, y_score),
        "pr_auc": metrics.average_precision_score(y_true, y_score),
    }


def plot_confusion(y_true, y_pred, title, ax):
    """2x2 confusion matrix with absolute counts; rows are truth, columns predictions."""
    cm = metrics.confusion_matrix(y_true, y_pred)
    disp = metrics.ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"])
    disp.plot(ax=ax, colorbar=False, values_format=",d")
    ax.grid(False)  # whitegrid lines strike through the count labels otherwise
    ax.set_title(title)
    return ax
