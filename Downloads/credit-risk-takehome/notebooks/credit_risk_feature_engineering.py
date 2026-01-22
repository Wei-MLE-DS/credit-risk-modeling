from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve,
    precision_recall_curve, 
    average_precision_score,
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score, 
    average_precision_score
)
def check_missing_values(df: pd.DataFrame, missing_value_rules: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Check for missing values using standard NA detection and missing value rules.
    Compute the count and percentage of missing values for each column
    Parameters: 
    df: pd.DataFrame, 
    missing_value_rules: dict, optional, a dictionary of column -> sentinel missing value mappings. 
    Example:  
    missing_value_rules = {
        "fico_score": 99999,
        "income": -1,
        "inq_last_6m": 99 
    }
    If None, the function requires to pass own rules using a separate function 
    Returns:
    pd.DataFrame 
        Summary DataFrame with columns:
        - column: column name
        - missing_count: total number of missing values
        - missing_pct: percentage of missing values (0-100) 
        
    Note: 
    - The function does not modify the original DataFrame. 
    - Columns missing from missing_value_rules are evaluated only using standard NA detection. 
    - Column rules treat sentinel values as missing *in addition* to the standard NA detection. 
    """
    if missing_value_rules is None:
        missing_value_rules = {}
    summary_rows = []
    n = len(df)
    for col in df.columns:
        # start with standard pandas missing values
        missing_mask = df[col].isna()
        # add missing_value_rules if applicable
        if col in missing_value_rules:
            sentinel_mask = missing_value_rules[col]
            missing_mask |= (df[col] == sentinel_mask)
        missing_count = missing_mask.sum()
        missing_pct = (missing_count / n * 100) if n > 0 else 0.0 
        summary_rows.append({
            "column": col,
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2)
        })
    return pd.DataFrame(summary_rows)

def plot_discrete_distribution(
    df: Union[pd.DataFrame, pd.Series],
    col: str,
    missing_sentinel: Union[int, float] = None,
    title: str = None,
    palette: str = "viridis",
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plots the distribution of a discrete/integer column using a bar chart.
    Treats missing_sentinel as 'Missing'.
    """
    if isinstance(df, pd.DataFrame):
        s = df[col]
    else:
        s = pd.Series(df)
        
    total_n = len(s)
    
    # Identify missing values vs valid data
    missing_mask = s.isna() | (s == missing_sentinel) if missing_sentinel is not None else s.isna()
    valid_data = s[~missing_mask]
    missing_count = int(missing_mask.sum())
    
    # Calculate counts and format labels
    counts = valid_data.value_counts().sort_index()
    labels = [str(int(i)) if i == int(i) else str(i) for i in counts.index]
    
    plot_data = []
    for i, label in enumerate(labels):
        count = counts.iloc[i]
        plot_data.append({"Category": label, "Count": count, "Pct": (count / total_n) * 100})
    
    # Add Missing group
    plot_data.append({"Category": "Missing", "Count": missing_count, "Pct": (missing_count / total_n) * 100})
    plot_df = pd.DataFrame(plot_data)
    
    # Visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set colors: Palette for data, Grey for Missing
    colors = list(sns.color_palette(palette, n_colors=len(labels))) + [(0.5, 0.5, 0.5)]
    sns.barplot(data=plot_df, x="Category", y="Count", palette=colors, ax=ax)
    
    ax.set_title(title or f"{col.replace('_', ' ').title()} Distribution", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel("Account Count", fontsize=12)
    
    # Annotate bars with percentages
    for i, p in enumerate(ax.patches):
        pct = plot_df.iloc[i]["Pct"]
        ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return ax

def plot_continuous_distribution_hist(
    df: Union[pd.DataFrame, pd.Series],
    col: str,
    title: str = None,
    color: str = "skyblue",
    bins: int = 30,
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plots the distribution of continuous data using a histogram and KDE.
    Includes a stats box with Mean, Median, and Missing counts.
    """
    if isinstance(df, pd.DataFrame):
        s = df[col]
    else:
        s = pd.Series(df)
        
    valid_data = s.dropna()
    total_n = len(s)
    missing_count = total_n - len(valid_data)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(valid_data, kde=True, color=color, ax=ax, bins=bins, edgecolor="white")
    
    ax.set_title(title or f"{col.replace('_', ' ').title()} Histogram Distribution", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    # Information Box
    stats_text = (
        f"Mean: {valid_data.mean():.2f}\n"
        f"Median: {valid_data.median():.2f}\n"
        f"Std Dev: {valid_data.std():.2f}\n"
        f"Missing: {missing_count} ({(missing_count/total_n)*100:.1f}%)"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
            fontsize=10)
            
    plt.tight_layout()
    return ax

def plot_fico_score_distribution(
    df: pd.DataFrame,
    fico_col: str = "fico_score",
    missing_sentinel: int = 99999,
    title: str = "FICO Score Distribution",
    palette: str = "Blues_d",
    figsize: tuple = (10, 5)) -> plt.Axes:
    """
    Plot the distribution of FICO scores directly and return the matplotlib Axes.
    Assumptions: 
    - Valid FICO values are in [550, 850]
    - When FICO value is 99999, it will be considered as missing
    
    Parameters:
    df: pd.DataFrame
    fico_col: str, default "fico_score"
    missing_setinel: int, default 99999
    title: str, defaults "FICO Score Distribution"
    palette: str, defaults "Blues_d"
    figsize: tuple, default (10, 5)

    Returns:
    matplotlib.axes.Axes
    """
    bins = [550, 580, 620, 660, 690, 720, 760, 850]
    labels = ["550-579", "580-619", "620-659", "660-689", "690-719", "720-759", "760-850"]
    if isinstance(df, pd.DataFrame):
        s = df[fico_col]
    else:
        s = pd.Series(df)
    total_n = len(s)
    # missing mask: sentinel
    missing_mask = s.isna() | (s == missing_sentinel)
    # Bin valid values
    binned = pd.cut(
        s.mask(missing_mask),
        bins = bins,
        labels = labels,
        right = True,
        include_lowest = True
    )
    # count valid bins - ensure all labels are included even if count is 0
    counts = binned.value_counts(dropna=False).sort_index()
    # Create a Series with all labels initialized to 0
    all_counts = pd.Series(0, index=labels)
    # Update with actual counts
    all_counts.update(counts)
    # Convert to DataFrame
    counts_df = all_counts.rename("count").to_frame()
    
    # Remove any NaN index entries if they exist
    if pd.isna(counts_df.index).any():
        counts_df = counts_df[counts_df.index.notna()]
    
    # add missing category explicitly
    missing_count = int(missing_mask.sum())
    counts_df.loc["Missing"] = missing_count 
    
    # compute percentage
    counts_df["pct"] = (counts_df["count"] / total_n * 100).round(2)
    
    # Order bins + missing
    desired_order = labels + ["Missing"]
    counts_df = counts_df.reindex(desired_order)
    
    # prepare for seaborn
    plot_df = counts_df.reset_index().rename(columns = {"index" : "bin"})
    
    # plot
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data = plot_df,
        x = "bin", 
        y = "count", 
        palette = "Blues_d"
    )
    ax.set_title("FICO Score Distribution")
    ax.set_xlabel("FICO Score Bin")
    ax.set_ylabel("Count")
    for p, pct in zip(ax.patches, plot_df["pct"]): 
        ax.annotate(
            f"{pct:.2f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha = "center",
            va = "bottom",
            fontsize = 8
        )
    plt.tight_layout()
    return ax

def plot_income_distribution(
    df: Union[pd.DataFrame, pd.Series],
    income_col: str = "income",
    missing_sentinel: int = -1,
    title: str = "Income Distribution",
    palette: str = "Blues_d",
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plot the distribution of Income directly and return the matplotlib Axes.
    Assumptions: 
    - Valid Income values are >= 0
    - When Income value is -1, it will be considered as missing
    
    Parameters:
    -----------
    df: pd.DataFrame or pd.Series
    income_col: str, default "income"
    missing_sentinel: int, default -1
    title: str, defaults "Income Distribution"
    palette: str, defaults "Blues_d"
    figsize: tuple, default (10, 5)

    Returns:
    --------
    matplotlib.axes.Axes
    """
    # Define Income Bins and Labels
    # Ranges: Under 30k, 30-60k, 60-90k, 90-120k, 120-150k, 150-250k, 250k+
    bins = [0, 30000, 60000, 90000, 120000, 150000, 250000, 1000000]
    labels = ["<30k", "30k-60k", "60k-90k", "90k-120k", "120k-150k", "150k-250k", "250k+"]
    
    if isinstance(df, pd.DataFrame):
        s = df[income_col]
    else:
        s = pd.Series(df)
        
    total_n = len(s)
    
    # Missing mask: sentinel (-1) or standard NaNs
    missing_mask = s.isna() | (s == missing_sentinel)
    
    # Bin valid values (right=False means 30,000 falls into the "30k-60k" bucket)
    binned = pd.cut(
        s.mask(missing_mask),
        bins = bins,
        labels = labels,
        right = False,
        include_lowest = True
    )
    
    # Count valid bins - ensure all labels are included even if count is 0
    counts = binned.value_counts(dropna=False).sort_index()
    all_counts = pd.Series(0, index=labels)
    all_counts.update(counts)
    
    # Convert to DataFrame and remove any auxiliary NaN index
    counts_df = all_counts.rename("count").to_frame()
    if pd.isna(counts_df.index).any():
        counts_df = counts_df[counts_df.index.notna()]
    
    # Add missing category explicitly
    missing_count = int(missing_mask.sum())
    counts_df.loc["Missing"] = missing_count 
    
    # Compute percentage
    counts_df["pct"] = (counts_df["count"] / total_n * 100).round(2)
    
    # Ensure order: Labels first, then Missing
    desired_order = labels + ["Missing"]
    counts_df = counts_df.reindex(desired_order)
    
    # Prepare for plotting
    plot_df = counts_df.reset_index().rename(columns = {"index" : "bin"})
    
    # Plotting
    plt.figure(figsize=figsize)
    # Valid bins use your palette; 'Missing' is assigned a contrasting red
    base_colors = sns.color_palette(palette, n_colors=len(labels))
    final_colors = list(base_colors) + [(0.91, 0.30, 0.24)] 
    
    ax = sns.barplot(data=plot_df, x="bin", y="count", palette=final_colors)
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Income Range", fontsize=12)
    ax.set_ylabel("Account Count", fontsize=12)
    
    # Annotate bars with percentages
    for i, p in enumerate(ax.patches):
        pct = plot_df.iloc[i]["pct"]
        ax.annotate(
            f"{pct:.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha = "center", va = "bottom",
            fontsize = 10, fontweight='bold',
            xytext=(0, 5), textcoords='offset points'
        )
        
    plt.tight_layout()
    return ax


def plot_employment_length_distribution(
    df: Union[pd.DataFrame, pd.Series],
    emp_col: str = "employment_length",
    missing_sentinel: int = -1,
    title: str = "Employment Length Distribution",
    palette: str = "Blues_d",
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plot the distribution of Employment Length directly and return the matplotlib Axes.
    Assumptions: 
    - Valid values are >= 0 (representing years of employment).
    - When value matches missing_sentinel, it will be considered as missing.
    
    Parameters:
    -----------
    df: pd.DataFrame or pd.Series
    emp_col: str, default "employment_length"
    missing_sentinel: int, default -1
    title: str, defaults "Employment Length Distribution"
    palette: str, defaults "Greens_d"
    figsize: tuple, default (12, 6)

    Returns:
    --------
    matplotlib.axes.Axes
    """
    # Industry standard buckets for employment stability
    bins = [0, 2, 5, 10, 20, 100]
    labels = ["0-1", "2-4", "5-9", "10-19", "20+"]
    
    if isinstance(df, pd.DataFrame):
        s = df[emp_col]
    else:
        s = pd.Series(df)
        
    total_n = len(s)
    missing_mask = s.isna() | (s == missing_sentinel)
    
    # Binning with right=False (e.g., 2 years falls into the 2-4 bucket)
    binned = pd.cut(
        s.mask(missing_mask),
        bins = bins,
        labels = labels,
        right = False
    )
    
    counts = binned.value_counts(dropna=False).sort_index()
    all_counts = pd.Series(0, index=labels)
    all_counts.update(counts)
    counts_df = all_counts.rename("count").to_frame()
    
    if pd.isna(counts_df.index).any():
        counts_df = counts_df[counts_df.index.notna()]
    
    counts_df.loc["Missing"] = int(missing_mask.sum()) 
    counts_df["pct"] = (counts_df["count"] / total_n * 100).round(2)
    
    plot_df = counts_df.reindex(labels + ["Missing"]).reset_index().rename(columns = {"index" : "bin"})
    
    plt.figure(figsize=figsize)
    base_colors = sns.color_palette(palette, n_colors=len(labels))
    final_colors = list(base_colors) + [(0.91, 0.30, 0.24)] # Red for Missing
    
    ax = sns.barplot(data = plot_df, x = "bin", y = "count", palette = final_colors)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Years of Employment", fontsize=12)
    ax.set_ylabel("Account Count", fontsize=12)
    
    for i, p in enumerate(ax.patches):
        pct = plot_df.iloc[i]["pct"]
        ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    return ax


def plot_age_distribution(
    df: Union[pd.DataFrame, pd.Series],
    age_col: str = "age",
    missing_sentinel: int = -1,
    title: str = "Age Distribution",
    palette: str = "Blues_d",
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plot the distribution of Age directly and return the matplotlib Axes.
    Assumptions: 
    - Valid values are typically >= 18.
    - When value matches missing_sentinel, it will be considered as missing.
    
    Parameters:
    -----------
    df: pd.DataFrame or pd.Series
    age_col: str, default "age"
    missing_sentinel: int, default -1
    title: str, defaults "Age Distribution"
    palette: str, defaults "Purples_d"
    figsize: tuple, default (12, 6)

    Returns:
    --------
    matplotlib.axes.Axes
    """
    # Standard demographic age buckets
    bins = [0, 25, 35, 45, 55, 65, 150]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    
    if isinstance(df, pd.DataFrame):
        s = df[age_col]
    else:
        s = pd.Series(df)
        
    total_n = len(s)
    missing_mask = s.isna() | (s == missing_sentinel)
    
    binned = pd.cut(
        s.mask(missing_mask),
        bins = bins,
        labels = labels,
        right = False
    )
    
    counts = binned.value_counts(dropna=False).sort_index()
    all_counts = pd.Series(0, index=labels)
    all_counts.update(counts)
    counts_df = all_counts.rename("count").to_frame()
    
    if pd.isna(counts_df.index).any():
        counts_df = counts_df[counts_df.index.notna()]
    
    counts_df.loc["Missing"] = int(missing_mask.sum()) 
    counts_df["pct"] = (counts_df["count"] / total_n * 100).round(2)
    
    plot_df = counts_df.reindex(labels + ["Missing"]).reset_index().rename(columns = {"index" : "bin"})
    
    plt.figure(figsize=figsize)
    base_colors = sns.color_palette(palette, n_colors=len(labels))
    final_colors = list(base_colors) + [(0.91, 0.30, 0.24)] # Red for Missing
    
    ax = sns.barplot(data = plot_df, x = "bin", y = "count", palette = final_colors)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Age Groups", fontsize=12)
    ax.set_ylabel("Account Count", fontsize=12)
    
    for i, p in enumerate(ax.patches):
        pct = plot_df.iloc[i]["pct"]
        ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    return ax


def plot_float_distribution_hist(
    df: Union[pd.DataFrame, pd.Series],
    col: str,
    missing_sentinel: float = -1.0,
    title: str = "Distribution Plot",
    color: str = "skyblue",
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plot the continuous distribution of a float variable using a histogram and KDE.
    Missing values (sentinels or NaNs) are excluded from the plot but reported in stats.

    Parameters:
    -----------
    df: pd.DataFrame or pd.Series
    col: str
        Column name to plot (e.g., 'debt_to_income').
    missing_sentinel: float, default -1.0
        Value representing missing data.
    title: str, defaults "Distribution Plot"
    color: str, defaults "skyblue"
    figsize: tuple, default (12, 6)

    Returns:
    --------
    matplotlib.axes.Axes
    """
    if isinstance(df, pd.DataFrame):
        s = df[col]
    else:
        s = pd.Series(df)
        
    total_n = len(s)
    # Mask sentinels and NaNs
    missing_mask = s.isna() | (s == missing_sentinel)
    valid_data = s[~missing_mask]
    
    missing_count = int(missing_mask.sum())
    missing_pct = (missing_count / total_n) * 100
    
    # Set the visual style
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram with Kernel Density Estimate (KDE)
    sns.histplot(
        valid_data, 
        kde=True, 
        color=color, 
        ax=ax, 
        bins=30, 
        edgecolor="white",
        line_kws={"linewidth": 2.5}
    )
    
    # Polish formatting
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(col.replace("_", " ").title(), fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    
    # Add an information box for stats and data quality
    stats_text = (
        f"Mean: {valid_data.mean():.3f}\n"
        f"Median: {valid_data.median():.3f}\n"
        f"Std Dev: {valid_data.std():.3f}\n"
        f"Missing: {missing_count} ({missing_pct:.1f}%)"
    )
    
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
            fontsize=10)
            
    plt.tight_layout()
    return ax


def plot_channel_distribution(
    df: Union[pd.DataFrame, pd.Series],
    channel_col: str = "channel",
    title: str = "Loan Channel Distribution",
    palette: str = "viridis",
    figsize: tuple = (10, 6)
) -> plt.Axes:
    """
    Plots the distribution of Loan Channels, handling missing values and 
    reporting percentages on each bar.
    """
    if isinstance(df, pd.DataFrame):
        s = df[channel_col]
    else:
        s = pd.Series(df)
    
    total_n = len(s)
    missing_count = int(s.isna().sum())
    
    # Calculate counts for existing categories
    counts = s.dropna().value_counts().sort_index()
    labels = counts.index.tolist()
    
    # Consolidate data for plotting
    plot_data = []
    for label in labels:
        count = counts[label]
        plot_data.append({"Category": label, "Count": count, "Pct": (count / total_n) * 100})
    
    # Append the "Missing" category
    plot_data.append({"Category": "Missing", "Count": missing_count, "Pct": (missing_count / total_n) * 100})
    plot_df = pd.DataFrame(plot_data)

    # Visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set colors: Paletted for categories, Grey for Missing
    base_colors = sns.color_palette(palette, n_colors=len(labels))
    final_colors = list(base_colors) + [(0.5, 0.5, 0.5)] 
    
    sns.barplot(data=plot_df, x="Category", y="Count", palette=final_colors, ax=ax)
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Origination Channel", fontsize=12)
    ax.set_ylabel("Account Count", fontsize=12)
    
    # Percentage Annotations
    for i, p in enumerate(ax.patches):
        pct = plot_df.iloc[i]["Pct"]
        ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return ax

def plot_product_type_distribution(
    df: Union[pd.DataFrame, pd.Series],
    product_col: str = "product_type",
    title: str = "Product Type Distribution",
    palette: str = "rocket",
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plots the distribution of Loan Product Types, handling missing values.
    """
    if isinstance(df, pd.DataFrame):
        s = df[product_col]
    else:
        s = pd.Series(df)
    
    total_n = len(s)
    missing_count = int(s.isna().sum())
    
    # Counts for non-missing
    counts = s.dropna().value_counts().sort_index()
    labels = counts.index.tolist()
    
    # Build the final data for plotting
    plot_data = []
    for label in labels:
        count = counts[label]
        pct = (count / total_n) * 100
        plot_data.append({"Category": label, "Count": count, "Pct": pct})
    
    # Add Missing
    plot_data.append({"Category": "Missing", "Count": missing_count, "Pct": (missing_count / total_n) * 100})
    
    plot_df = pd.DataFrame(plot_data)

    # Visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    base_colors = sns.color_palette(palette, n_colors=len(labels))
    final_colors = list(base_colors) + [(0.5, 0.5, 0.5)] # Grey for Missing
    
    sns.barplot(data=plot_df, x="Category", y="Count", palette=final_colors, ax=ax)
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Loan Product", fontsize=12)
    ax.set_ylabel("Account Count", fontsize=12)
    
    for i, p in enumerate(ax.patches):
        pct = plot_df.iloc[i]["Pct"]
        ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return ax

def plot_state_distribution(
    df: Union[pd.DataFrame, pd.Series],
    state_col: str = "state",
    title: str = "Distribution of Loans by State",
    palette: str = "coolwarm",
    figsize: tuple = (12, 6)
) -> plt.Axes:
    """
    Plots the distribution of Loans by State, handling missing values and 
    displaying percentages.
    """
    if isinstance(df, pd.DataFrame):
        s = df[state_col]
    else:
        s = pd.Series(df)
    
    total_n = len(s)
    missing_count = int(s.isna().sum())
    
    # Calculate counts and sort by frequency (descending)
    counts = s.dropna().value_counts()
    labels = counts.index.tolist()
    
    # Consolidate data for plotting
    plot_data = []
    for label in labels:
        count = counts[label]
        plot_data.append({"State": label, "Count": count, "Pct": (count / total_n) * 100})
    
    # Append the "Missing" category if any
    if missing_count > 0:
        plot_data.append({"State": "Missing", "Count": missing_count, "Pct": (missing_count / total_n) * 100})
    
    plot_df = pd.DataFrame(plot_data)

    # Visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use palette for states, grey for Missing if it exists
    n_states = len(labels)
    base_colors = sns.color_palette(palette, n_colors=n_states)
    final_colors = list(base_colors)
    if missing_count > 0:
        final_colors.append((0.5, 0.5, 0.5))
    
    sns.barplot(data=plot_df, x="State", y="Count", palette=final_colors, ax=ax)
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("State", fontsize=12)
    ax.set_ylabel("Account Count", fontsize=12)
    
    # Percentage Annotations
    for i, p in enumerate(ax.patches):
        pct = plot_df.iloc[i]["Pct"]
        ax.annotate(f"{pct:.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', xytext=(0, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return ax

def detect_outliers_iqr(df: pd.DataFrame, col: str, lower_percentile: float = 0.25, 
                        upper_percentile: float = 0.75, multiplier: float = 1.5) -> dict:
    """
    Detect outliers using IQR method.
    Returns dictionary with outlier statistics.
    Note: Missing values are excluded from outlier calculation but reported separately.
    """
    # Handle missing sentinels
    df_clean = df.copy()
    if col == 'fico_score':
        df_clean[col] = df_clean[col].replace(99999, np.nan)
    elif col == 'inquiries_last_6m':
        df_clean[col] = df_clean[col].replace(99, np.nan)
    elif col == 'income':
        df_clean[col] = df_clean[col].replace(-1, np.nan)
    
    # Calculate on non-missing values only
    valid_data = df_clean[col].dropna()
    missing_count = df_clean[col].isna().sum()
    
    if len(valid_data) == 0:
        return {
            'column': col,
            'missing_count': missing_count,
            'missing_pct': (missing_count / len(df)) * 100,
            'outlier_count': 0,
            'outlier_pct': 0.0,
            'note': 'All values missing'
        }
    
    Q1 = valid_data.quantile(lower_percentile)
    Q3 = valid_data.quantile(upper_percentile)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = valid_data[(valid_data < lower_bound) | (valid_data > upper_bound)]
    outlier_pct = (len(outliers) / len(valid_data)) * 100 if len(valid_data) > 0 else 0
    
    return {
        'column': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'valid_count': len(valid_data),
        'missing_count': missing_count,
        'missing_pct': (missing_count / len(df)) * 100,
        'outlier_count': len(outliers),
        'outlier_pct': outlier_pct,
        'min_value': valid_data.min(),
        'max_value': valid_data.max()
    }

def plot_outlier_boxplot(df: pd.DataFrame, col: str, figsize: tuple = (8, 6)) -> plt.Axes:
    """Plot boxplot to visualize outliers (missing values shown separately)"""
    # Handle missing sentinels
    df_plot = df.copy()
    if col == 'fico_score':
        df_plot[col] = df_plot[col].replace(99999, np.nan)
    elif col == 'inquiries_last_6m':
        df_plot[col] = df_plot[col].replace(99, np.nan)
    elif col == 'income':
        df_plot[col] = df_plot[col].replace(-1, np.nan)
    
    missing_count = df_plot[col].isna().sum()
    
    plt.figure(figsize=figsize)
    ax = sns.boxplot(y=df_plot[col])
    title = f"Boxplot: {col.replace('_', ' ').title()}"
    if missing_count > 0:
        title += f"\n(Missing: {missing_count}, {missing_count/len(df)*100:.1f}%)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(col.replace('_', ' ').title())
    plt.tight_layout()
    return ax

def plot_default_rate_by_feature(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str = "default_12m",
    title: str = None,
    figsize: tuple = (12, 6),
    top_n: int = None,
    include_missing: bool = True
) -> plt.Axes:
    """
    Plot default rate across feature levels with sample sizes.
    Missing values are included as a separate category if include_missing=True.
    """
    # Handle missing sentinels
    df_plot = df.copy()
    if feature_col == 'fico_score':
        df_plot[feature_col] = df_plot[feature_col].replace(99999, np.nan)
    elif feature_col == 'inquiries_last_6m':
        df_plot[feature_col] = df_plot[feature_col].replace(99, np.nan)
    elif feature_col == 'income':
        df_plot[feature_col] = df_plot[feature_col].replace(-1, np.nan)
    
    # Group by feature (including missing as a category)
    if include_missing:
        # Convert to string/object type if it's categorical to allow adding 'Missing'
        if pd.api.types.is_categorical_dtype(df_plot[feature_col]):
            df_plot[feature_col] = df_plot[feature_col].astype(str)
        # Replace NaN with 'Missing' for grouping
        df_plot[feature_col] = df_plot[feature_col].fillna('Missing')
    
    grouped = df_plot.groupby(feature_col)[target_col].agg(['mean', 'count'])
    grouped.columns = ['default_rate', 'count']
    grouped = grouped.sort_values('default_rate', ascending=False)
    
    # Limit to top N if specified (but always include 'Missing' if it exists)
    if top_n is not None and len(grouped) > top_n:
        if 'Missing' in grouped.index:
            # Keep Missing and top N-1 others
            missing_row = grouped.loc[['Missing']]
            others = grouped.drop('Missing').head(top_n - 1)
            grouped = pd.concat([missing_row, others]).sort_values('default_rate', ascending=False)
        else:
            grouped = grouped.head(top_n)
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Default rate bars (use different color for Missing)
    colors = ['coral' if str(idx) != 'Missing' else 'gray' for idx in grouped.index]
    bars = ax1.bar(range(len(grouped)), grouped['default_rate'] * 100, 
                   color=colors, alpha=0.7, label='Default Rate')
    ax1.set_ylabel('Default Rate (%)', color='coral', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='coral')
    ax1.set_xticks(range(len(grouped)))
    ax1.set_xticklabels([str(x) for x in grouped.index], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(grouped['default_rate'] * 100) * 1.2)
    
    # Count line on secondary axis
    ax2 = ax1.twinx()
    line = ax2.plot(range(len(grouped)), grouped['count'], 
                    color='steelblue', marker='o', linewidth=2.5, 
                    markersize=8, label='Loan Count')
    ax2.set_ylabel('Loan Count', color='steelblue', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    
    # Add count annotations on bars
    for i, (idx, row) in enumerate(grouped.iterrows()):
        ax1.text(i, row['default_rate'] * 100, f"n={int(row['count'])}",
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax1.set_title(title or f"Default Rate by {feature_col.replace('_', ' ').title()}", 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel(feature_col.replace('_', ' ').title(), fontsize=12)
    
    plt.tight_layout()
    return ax1

def check_class_distribution(y, dataset_name):
    """Check and display class distribution"""
    total = len(y)
    class_counts = y.value_counts().sort_index()
    class_pct = (class_counts / total * 100).round(2)
    
    print(f"\n{dataset_name} Class Distribution:")
    print(f"  Total samples: {total}")
    print(f"  Class 0 (No Default): {class_counts.get(0, 0)} ({class_pct.get(0, 0):.2f}%)")
    print(f"  Class 1 (Default): {class_counts.get(1, 0)} ({class_pct.get(1, 0):.2f}%)")
    print(f"  Imbalance ratio: {class_counts.get(0, 1) / class_counts.get(1, 1):.2f}:1")
    return class_counts, class_pct

def identify_missing_columns(df, missing_rules):
    """Replace sentinels with NaN and identify columns with missing values"""
    df_check = df.copy()
    if 'fico_score' in df_check.columns:
        df_check['fico_score'] = df_check['fico_score'].replace(99999, np.nan)
    if 'inquiries_last_6m' in df_check.columns:
        df_check['inquiries_last_6m'] = df_check['inquiries_last_6m'].replace(99, np.nan)
    if 'income' in df_check.columns:
        df_check['income'] = df_check['income'].replace(-1, np.nan)
    
    missing_cols = df_check.columns[df_check.isna().any()].tolist()
    return missing_cols, df_check

def replace_sentinels(df):
    """Replace sentinel values with NaN"""
    df_clean = df.copy()
    if 'fico_score' in df_clean.columns:
        df_clean['fico_score'] = df_clean['fico_score'].replace(99999, np.nan)
    if 'inquiries_last_6m' in df_clean.columns:
        df_clean['inquiries_last_6m'] = df_clean['inquiries_last_6m'].replace(99, np.nan)
    if 'income' in df_clean.columns:
        df_clean['income'] = df_clean['income'].replace(-1, np.nan)
    return df_clean

def impute_missing(df, numeric_imputer, categorical_imputer, 
                   numeric_cols, categorical_cols):
    """Apply fitted imputers to a dataset"""
    df_imputed = df.copy()
    
    if numeric_cols:
        df_imputed[numeric_cols] = numeric_imputer.transform(df_imputed[numeric_cols])
    if categorical_cols:
        df_imputed[categorical_cols] = categorical_imputer.transform(df_imputed[categorical_cols])
    
    return df_imputed

def detect_outlier_bounds(df, col, method='iqr', multiplier=1.5):
    """
    Detect outlier bounds using IQR method.
    Returns lower and upper bounds.
    """
    if method == 'iqr':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
    elif method == 'zscore':
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    return lower_bound, upper_bound

def create_fico_bins(df, bins, labels):
    """Create FICO bins"""
    df_binned = df.copy()
    df_binned['fico_binned'] = pd.cut(df_binned['fico_score'], 
                                      bins=bins, labels=labels, 
                                      right=True, include_lowest=True)
    return df_binned

def create_income_bins(df, bins, labels):
    """Create income bins"""
    df_binned = df.copy()
    df_binned['income_binned'] = pd.cut(df_binned['income'], 
                                        bins=bins, labels=labels, 
                                        right=False, include_lowest=True)
    return df_binned

def ks_statistic(y_true, y_score):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_score": y_score
    }).sort_values("y_score", ascending=False)

    df["cum_event"] = (df["y_true"] == 1).cumsum() / (df["y_true"] == 1).sum()
    df["cum_non_event"] = (df["y_true"] == 0).cumsum() / (df["y_true"] == 0).sum()

    ks = np.max(np.abs(df["cum_event"] - df["cum_non_event"]))
    return ks
def evaluate_binary_model(y_true, y_score, dataset_name=""):
    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    ks = ks_statistic(y_true, y_score)

    print(f"{dataset_name}")
    print("-" * 40)
    print(f"AUC-ROC      : {auc:.4f}")
    print(f"PR AUC       : {pr_auc:.4f}")
    print(f"KS Statistic : {ks:.4f}")
    print()

def plot_roc_pr(y_true, y_score, dataset_name=""):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {dataset_name}")

    # PR
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve – {dataset_name}")

    plt.tight_layout()
    plt.show()

def find_best_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]

def evaluate_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print()
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=4))

def ks_with_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks_values = np.abs(tpr - fpr)
    idx = np.argmax(ks_values)
    return ks_values[idx], thresholds[idx]

def find_best_ks_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = tpr - fpr
    idx = np.argmax(ks)
    return thresholds[idx], ks[idx]

def calculate_ks_statistic(y_true, y_score):
    """Calculate Kolmogorov-Smirnov statistic"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    return ks

def calculate_all_metrics(y_true, y_score, y_pred, dataset_name, model_name):
    """Calculate comprehensive metrics for a model"""
    metrics = {
        'Model': model_name,
        'Dataset': dataset_name,
        'AUC-ROC': roc_auc_score(y_true, y_score),
        'PR-AUC': average_precision_score(y_true, y_score),
        'KS': calculate_ks_statistic(y_true, y_score),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics

# Interpret key drivers
def feature_direction(shap_values, X):
    """
    Get feature direction: whether high values increase or decrease risk.
    
    Parameters:
        shap_values: np.ndarray
        X: pd.DataFrame
    
    Returns:
        direction_df: pd.DataFrame, positive mean SHAP -> increases default
    """
    mean_shap = pd.DataFrame(shap_values, columns=X.columns).mean()
    direction_df = pd.DataFrame({
        "feature": mean_shap.index,
        "mean_shap": mean_shap.values,
        "direction": ["increase_risk" if v > 0 else "decrease_risk" for v in mean_shap.values]
    }).sort_values("mean_shap", key=abs, ascending=False)
    
    return direction_df
    
def plot_dependence_feature(shap_values, X, feature_name):
    """
    Plot SHAP dependence for a single feature.
    
    Parameters:
        shap_values: np.ndarray
        X: pd.DataFrame
        feature_name: str, feature to plot
    """
    shap.dependence_plot(feature_name, shap_values, X)

def summarize_shap(shap_values, X, top_n=10):
    """
    Generate SHAP importance summary table.
    
    Parameters:
        shap_values: np.ndarray, SHAP values
        X: pd.DataFrame, original features
        top_n: int, number of top features to return
    
    Returns:
        importance_df: pd.DataFrame, features ranked by mean absolute SHAP
    """
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "shap_importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)
    
    return importance_df.head(top_n)