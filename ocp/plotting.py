import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}

plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


def plot_barplot(figsize=(16, 10), title='',
                 xlabel=None, ylabel=None,
                 xticks=None, yticks=None,
                 xticks_labels=None, yticks_labels=None,
                 xlabel_size=8, ylabel_size=10,
                 xticks_size=8, yticks_size=10,
                 xlim=None, ylim=None,
                 **barplot_params):
    plt.figure(figsize=figsize);
    sns.barplot(**barplot_params);
    plt.title(title);
    plt.xlabel(xlabel, fontsize=xlabel_size);
    plt.ylabel(ylabel, fontsize=ylabel_size);
    plt.xticks(ticks=xticks, labels=xticks_labels, size=xticks_size, rotation=90);
    plt.yticks(ticks=yticks, labels=yticks_labels, size=yticks_size);
    if xlim: plt.xlim(xlim);
    if ylim: plt.ylim(ylim);
    plt.grid()


def plot_missing_matrix(data, figsize=(22, 10), title='', **matrix_params):
    fig, ax = plt.subplots(figsize=figsize);
    msno.matrix(data, ax=ax, **matrix_params);
    ax.set_title(title);
    plt.tight_layout()


def plot_valid_curve(param_values,
                     train_scores_mean, train_scores_std,
                     test_scores_mean, test_scores_std,
                     figsize=(16, 8), title='',
                     xlabel='', ylim=None, ax=None):
    """Generate a simple plot of the test and training param curve"""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Score")

    # Plot curve
    ax.grid()

    ax.fill_between(param_values, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(param_values, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1,
                    color="g")

    ax.plot(param_values, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(param_values, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    ax.legend(loc="best")

    return plt
