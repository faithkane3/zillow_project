import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import numpy as np 


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
          df with new feature "tenure_years" as a category type
          calculating tenure in complete years
    """
    df["tenure_years"] = round(df.tenure // 12).astype("category")
    return df


def correlation_exploration(df, x_string, y_string):
    r, p = stats.pearsonr(df[x_string], df[y_string])
    df.plot.scatter(x_string, y_string)
    plt.title(f"{x_string}'s Relationship with {y_string}")
    print(f'The p-value is: {p}. There is {round(p,3)}% chance that we see these results by chance.')
    print(f'r = {round(r, 2)}')
    plt.show()


def plot_categorical_and_continuous_vars(df):
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
    
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind='box', palette='Greens')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="swarm", palette='Blues')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)
    
    sns.catplot(x=categorical_var, y=continuous_var, data=df, kind="bar", palette='Purples')
    plt.xlabel(categorical_var, fontsize=12)
    plt.ylabel(continuous_var, fontsize=12)


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