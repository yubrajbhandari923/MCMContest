import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def load_dataset():
    return pd.read_csv('Data/Wimbledon_featured_matches.csv')

def divide_by_matches(df=None):
    if df is None:
        df = load_dataset()
    return [df[df['match_id'] == i] for i in df['match_id'].unique()]

def divide_by_sets(match_df):
    return [match_df[match_df['set_no'] == i] for i in match_df['set_no'].unique()]

def divide_by_games(set_df):
    return [set_df[set_df['game_no'] == i] for i in set_df['game_no'].unique()]

def match_winner(match):
    last_row = match.iloc[-1]
    if last_row["p1_sets"] == last_row["p2_sets"]:
        return last_row["set_victor"]
    if last_row["p1_sets"] > last_row["p2_sets"]:
        return 1
    else:
        return 2
    
def add_match_victor(df):
    match_df = divide_by_matches(df)
    for match in match_df:
        winner = match_winner(match)
        match[:,"match_victor"] = winner
    return pd.concat(match_df)
    
    
def plot_column_vs_columns(match_df, col, cols, plot_f=sns.countplot):
    fig, ax = plt.subplots( len(cols)//2 + 2, 2, figsize=(15, 10))
    j = 0
    for i, column in enumerate(cols):
        plot_f(x=column, hue=col, data=match_df, ax=ax[j, i%2])
        plt.title(f'{column} vs. {col}')
        plt.xlabel(column)
        plt.ylabel('Count')
        fig.legend(title='Point Victor')
        if i % 2 == 1:
            j += 1
    plt.show()
    
def plot_match_flow(match_df):
    print(match_df.columns)
    match_df.loc[:,"p1_flow"] = match_df["p1_points_won"].cumsum()
    match_df.loc[:,"p2_flow"] = match_df["p2_points_won"].cumsum()
    sns.lineplot(x=match_df.index, y="p1_flow", data=match_df)
    sns.lineplot(x=match_df.index, y="p2_flow", data=match_df)
    plt.show()

