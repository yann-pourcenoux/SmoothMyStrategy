"""Module to analyse the performance of a model."""

import os

import pandas as pd
import quantstats_lumi as qs
import wandb

from constants import METRICS_REPORT_FILENAME


def log_report(dataframe: pd.DataFrame, output_path: str | None = None) -> None:
    """Logs the HTML to wand and saves it to a file."""

    report_path = os.path.join(output_path, METRICS_REPORT_FILENAME)
    qs.reports.html(returns=dataframe, output=report_path)
    with open(report_path) as html_report:
        wandb.log({"performance_report": wandb.Html(html_report)})
