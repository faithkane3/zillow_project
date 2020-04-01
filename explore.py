import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def plot_variable_pairs(df):
    """
    Takes:
          df
    Returns:
          PairGrid plot of all relationships
          histogram and scatter plots
    """
    g = sns.PairGrid(df)
    g.map_diag(plt.hist)
    g.map_offdiag(sns.regplot)
    plt.show()


def months_to_years(df):
    """
    Takes:
          df
    Returns:
          df with new feature "tenure_years"
    """
    df["tenure_years"] = round(df.tenure // 12).astype(object)
    return df


def plot_categorical_and_continuous_vars(categorical_var, continuous_var, df):
    """
    Takes:
          df
    Returns:
          three plots of categorical var with continuous var
    """
    plt.suptitle(f'{continuous_var} by {categorical_var}', fontsize=18)
    
    sns.lineplot(x=categorical_var, y=continuous_var, data=df)
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    plt.show()
    
    
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="swarm", palette='Blues')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    plt.show()
    
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="bar", palette='Purples')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    plt.show()


def plot_categorical_and_continuous_vars_telco(df):
    """
    Takes: 
        telco df
    Returns:
        three plots comparing tenure_years to total_charges
    """
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12,10), nrows=3,ncols=1, sharex=True)
    plt.style.use('seaborn-bright')

    plt.suptitle('Total Charges by Tenure Years', fontsize=18)

    ax1.plot(df.tenure_years, df.total_charges, color='mediumblue')
    ax1.set_ylabel('US Dollars', fontsize=14)

    ax2.bar(df.tenure_years, df.total_charges, color='dodgerblue')
    ax2.set_ylabel('US Dollars', fontsize=14)

    ax3.scatter(df.tenure_years, df.total_charges, color='skyblue')
    ax3.set_xlabel('Tenure in Years', fontsize=14)
    ax3.set_ylabel('US Dollars', fontsize=14)

    plt.tight_layout()
    plt.show()


def telco_pie(df):
    plt.style.use('seaborn-paper')
    labels = ['0 years', '1 years', '2 years', '3 years', '4 years', '5 years', '6 years']
    colors = ['dodgerblue', 'whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke', 'whitesmoke']
    explode = (0.1, 0, 0, 0, 0, 0, 0) 
    
    plt.pie(df.tenure_years.value_counts(), explode=explode, colors=colors, labels = labels, autopct='%1.0f%%', shadow=True, textprops={'fontsize':14}, wedgeprops={'edgecolor': 'black', 'width': 0.6})
    plt.title('Percent of Accounts by Tenure Years', fontsize=18)
    plt.show()


def correlation_exploration(df, x_string, y_string):
    r, p = stats.pearsonr(df[x_string], df[y_string])
    df.plot.scatter(x_string, y_string)
    plt.title(f"{x_string}'s Relationship with {y_string}")
    print(f'The p-value is: {p}. There is {round(p,3)}% chance that we see these results by chance.')
    print(f'r = {round(r, 2)}')
    plt.show()


def tax_distribution_viz(df):
    los_angeles_tax_dist = df[df.county_name == "Los Angeles"].tax_rate
    orange_tax_dist = df[df.county_name == "Orange"].tax_rate
    ventura_tax_dist = df[df.county_name == "Ventura"].tax_rate

    plt.figure(figsize=(16,14))

    plt.subplot(3,1,1)
    sns.distplot(los_angeles_tax_dist, bins=50, kde=True, rug=True)
    plt.xlim(0, .10)
    plt.ylim(0, 600)
    plt.title("Los Angeles County Tax Distribution")

    plt.subplot(3,1,2)
    sns.distplot(orange_tax_dist, bins=50, kde=True, rug=True, color='orange')
    plt.xlim(0, .10)
    plt.ylim(0, 600)
    plt.title("Orange County Tax Distribution")

    plt.subplot(3,1,3)
    sns.distplot(ventura_tax_dist, bins=50, kde=True, rug=True, color='green')
    plt.xlim(0, .10)
    plt.ylim(0, 600)
    plt.title("Ventura County Tax Distribution")

    plt.tight_layout()

    plt.show()