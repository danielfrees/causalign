"""
Visualizations and Data Analysis for the experiments data.

Loads in saved data from out/experiments.db and provides visualization functions.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname((__file__))), "out", "experiments.db"))

def load_data_from_db(table_name, db_path=DB_PATH):
    """
    Load data from a specified table in the SQLite database.
    
    Parameters:
    - table_name: str
        Name of the table to load data from.
    - db_path: str
        Path to the SQLite database file.

    Returns:
    - pandas.DataFrame
        Loaded data as a DataFrame.
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def filter_final_metrics(metrics_df):
    """
    Filter metrics DataFrame to only include rows where final metrics are available.
    
    Parameters:
    - metrics_df: pandas.DataFrame
        DataFrame containing metrics data.

    Returns:
    - pandas.DataFrame
        Filtered DataFrame.
    """
    return metrics_df.dropna(subset=["test_acc", "test_f1"])


def plot_final_metrics(metrics_df):
    """
    Plot final metrics (test accuracy and F1 score) across experiments as clustered bars.

    Parameters:
    - metrics_df: pandas.DataFrame
        DataFrame containing metrics data.
    """
    filtered_df = filter_final_metrics(metrics_df)
    
    experiment_ids = filtered_df["experiment_id"]
    test_acc = filtered_df["test_acc"]
    test_f1 = filtered_df["test_f1"]
    
    x = range(len(experiment_ids))
    bar_width = 0.35  # Width of each bar
    
    plt.figure(figsize=(12, 6))
    plt.bar(x, test_acc, width=bar_width, label="Test Accuracy", alpha=0.8)
    plt.bar([p + bar_width for p in x], test_f1, width=bar_width, label="Test F1 Score", alpha=0.8)
    plt.xlabel("Experiment ID")
    plt.ylabel("Metric Value")
    plt.title("Final Metrics Across Experiments")
    plt.xticks([p + bar_width / 2 for p in x], experiment_ids)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_metrics_vs_hyperparameters(arguments_df, 
                                    metrics_df, 
                                    hyperparameter_name, split="test"):
    """
    Plot final metrics (accuracy and F1 score) against a specified hyperparameter for a given split.

    Parameters:
    - arguments_df: pandas.DataFrame
        DataFrame containing arguments data.
    - metrics_df: pandas.DataFrame
        DataFrame containing metrics data.
    - hyperparameter_name: str
        Name of the hyperparameter to analyze.
    - split: str, default="test"
        The data split to analyze. One of ["train", "val", "test"].
    """
    if split not in ["train", "val", "test"]:
        raise ValueError("Invalid split. Must be one of ['train', 'val', 'test'].")
    acc_column = f"{split}_acc"
    f1_column = f"{split}_f1"

    if acc_column not in metrics_df.columns or f1_column not in metrics_df.columns:
        raise ValueError(f"Metrics DataFrame does not contain columns for '{split}' split.")

    merged_df = arguments_df.merge(metrics_df, left_on="id", right_on="experiment_id")
    merged_df = filter_final_metrics(merged_df)

    if f"arg_{hyperparameter_name}" not in merged_df.columns:
        raise ValueError(f"Hyperparameter '{hyperparameter_name}' not found in arguments table.")

    # Convert hyperparameter column to numeric if possible
    try:
        merged_df[f"arg_{hyperparameter_name}"] = pd.to_numeric(merged_df[f"arg_{hyperparameter_name}"])
    except ValueError:
        pass  # Leave as is if conversion fails

    plt.figure(figsize=(12, 6))
    plt.plot(merged_df[f"arg_{hyperparameter_name}"], merged_df[acc_column], label=f"{split.capitalize()} Accuracy", marker="o")
    plt.plot(merged_df[f"arg_{hyperparameter_name}"], merged_df[f1_column], label=f"{split.capitalize()} F1 Score", marker="o")
    plt.xlabel(hyperparameter_name)
    plt.ylabel("Metric Value")
    plt.title(f"{split.capitalize()} Metrics vs {hyperparameter_name}")
    plt.legend()
    plt.grid(True)
    plt.show()