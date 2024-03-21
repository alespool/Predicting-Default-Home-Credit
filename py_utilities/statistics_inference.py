import statsmodels.stats.api as sm
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_sample_size(effect_size, alpha, power, ratio=1):
    """
    Calculate the required sample size for a two-sample t-test.

    Parameters:
        effect_size: Desired effect size (Cohen's d).
        alpha: Significance level (e.g., 0.05).
        power: Desired statistical power (e.g., 0.80).
        ratio: Ratio of sample sizes in the two groups (default: 1).

    Returns:
        sample_size: Required sample size to achieve the desired power.
    """
    sample_size = sm.tt_ind_solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative="two-sided",
    )
    return round(sample_size)


def create_mean_std(group_a, group_b, column_name: str):
    """
    Calculates the mean and standard deviation for a specific column in two groups.

    Args:
        group_a (pandas.DataFrame): The first group.
        group_b (pandas.DataFrame): The second group.
        column_name (str): The name of the column to calculate the mean and standard deviation for.

    Returns:
        tuple: A tuple containing the mean and standard deviation for the column in group_a,
               followed by the mean and standard deviation for the column in group_b.
    """
    group_a_mean = group_a[column_name].mean()
    group_b_mean = group_b[column_name].mean()
    group_a_std = group_a[column_name].std()
    group_b_std = group_b[column_name].std()

    return group_a_mean, group_b_mean, group_a_std, group_b_std


def calculate_effect_size(
    group1_mean: float, group2_mean: float, group1_std: float, group2_std: float
):
    """
    Calculate Cohen's d effect size for a two-sample t-test.

    Parameters:
        group1_mean: Mean of the first group.
        group2_mean: Mean of the second group.
        group1_std: Standard deviation of the first group.
        group2_std: Standard deviation of the second group.

    Returns:
        effect_size: Cohen's d effect size.
    """
    pooled_std = ((group1_std**2) + (group2_std**2)) ** 0.5
    effect_size = abs(group1_mean - group2_mean) / pooled_std
    return effect_size


def test_normality_and_qq(data, column_name: str, alpha: float = 0.05):
    """
    Test the normality of a column using the Anderson-Darling test and create a QQ plot.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to test for normality.
        alpha (float): The significance level for the Anderson-Darling test (default is 0.05).

    Returns:
        bool: True if the data is normally distributed, False otherwise.
    """
    column_data = data[column_name].dropna()

    stat, p = stats.shapiro(column_data)

    is_normal = p > alpha
    return is_normal, p


def perform_bootstrapping(group_a, group_b, n_bootstrap_samples: int = 1000):
    """
    Perform bootstrapping to calculate differences in means between two groups.

    Parameters:
        group_a (numpy.ndarray or pandas.Series): Data from group A.
        group_b (numpy.ndarray or pandas.Series): Data from group B.
        n_bootstrap_samples (int): Number of bootstrap samples to generate.

    Returns:
        numpy.ndarray: Array containing the bootstrap differences in means.
    """
    bootstrap_diffs = []

    for _ in range(n_bootstrap_samples):
        sample_a = np.random.choice(group_a, len(group_a), replace=True)
        sample_b = np.random.choice(group_b, len(group_b), replace=True)

        bootstrap_diff = np.mean(sample_b) - np.mean(sample_a)
        bootstrap_diffs.append(bootstrap_diff)

    return np.array(bootstrap_diffs)

def bootstrap_ci(df, variable, classes, repetitions=1000, alpha=0.05, random_state=None):
    df = df[[variable, classes]]
    bootstrap_sample_size = len(df)

    mean_diffs = []
    for i in range(repetitions):
        bootstrap_sample = df.sample(n=bootstrap_sample_size, replace=True, random_state=random_state)
        mean_diff = bootstrap_sample.groupby(classes).mean().loc[:, variable].diff().iloc[-1]
        mean_diffs.append(mean_diff)
    
    # Confidence interval
    left = np.percentile(mean_diffs, (alpha/2) * 100)
    right = np.percentile(mean_diffs, (1 - alpha/2) * 100)
    
    # Point estimate
    point_est = df.groupby(classes).mean().loc[:, variable].diff().iloc[-1]
    
    print('Point estimate of difference between means:', round(point_est, 2))
    print((1-alpha)*100, '% confidence interval for the difference between means:', (round(left, 2), round(right, 2)), 'this is based on', repetitions, 'repetitions')

def missing_data(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)
    percent = percent.map('{:.2%}'.format)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data.head(50)