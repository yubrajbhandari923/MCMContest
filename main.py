import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.basicConfig(filename="prob.log", level=logging.DEBUG)

df = pd.read_csv('Data/Wimbledon_featured_matches.csv')

def divide_by_matches(df):
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

matches_df = divide_by_matches(df)


# global x
x =0
class Probablities:

    def __init__(self, Fij, state=(0, 0, 0, 0, 0, 0)):
        self.Fij = Fij
        self.state = state
        self.P_g_cache = {}
        self.P_s_cache = {}
        self.P_tb_cache = {}
        self.P_m_cache = {}

        global x
        x = 0

    def set_state(self, state):
        if len(state) != 6:
            raise ValueError("state should be a tuple or a list")

        si, sj, gi, gj, xi, xj = state
        map_score = {"0": 0, "15": 1, "30": 2, "40": 3}
        if xi in map_score and xj in map_score:
            self.state = (si, sj, gi, gj, map_score[xi], map_score[xj])
        elif xi == "AD":
            self.state = (si, sj, gi, gj, 3, 2)
        elif xj == "AD":
            self.state = (si, sj, gi, gj, 2, 3)
        else:
            self.state = (si, sj, gi, gj, xi, xj)
        return self.state

    def P_g(self, state):
        """
        P_g(xi, xj)
        """
        if len(state) != 6:
            raise ValueError("state should be a tuple or a list")

        si, sj, gi, gj, xi, xj = state

        as_str = "".join(map(str, state))

        # Check in cache
        if as_str in self.P_g_cache:
            return self.P_g_cache[as_str]

        if xi == 4 and xj <= 2:
            p = 1
        elif xj == 4 and xi <= 2:
            p = 0
        elif xi == 3 and xj == 3:
            p = (self.Fij**2) / (self.Fij**2 + (1 - self.Fij) ** 2)
        else:
            p = self.Fij * self.P_g((si, sj, gi, gj, xi + 1, xj)) + (
                1 - self.Fij
            ) * self.P_g((si, sj, gi, gj, xi, xj + 1))

        # print(f"P_g({state}) = {p}")
        self.P_g_cache[as_str] = p
        return p

    def P_s(self, state):
        """
        P_s(gi, gj)
        """
        if len(state) != 6:
            raise ValueError("state should be a tuple or a list")

        si, sj, gi, gj, xi, xj = state

        as_str = "".join(map(str, state))
        # Check in cache

        if as_str in self.P_s_cache:
            return self.P_s_cache[as_str]
        if gi >= 6 and (gi - gj) <= 4:
            p= 1
        elif gj >= 6 and (gj - gj) <= 4:
            p= 0
        elif gi == 6 and gj == 6:
            p= self.P_tb(state)
        else:
            p= self.P_g((si, sj, gi, gj, 0, 0)) * (
                1 - self.P_s((si, sj, gj, gi + 1, xi, xj))
            ) + (1 - self.P_g((si, sj, gi, gj, 0, 0))) * self.P_s(
                (si, sj, gj + 1, gi, xi, xj)
            )
        # print(f"P_s({state}) = {p}")

        # Store in cache
        self.P_s_cache[as_str] = p
        return p

    def P_tb(self, state):  # Ambiguous
        if len(state) != 6:
            raise ValueError("state should be a tuple or a list")
        raise NotImplementedError
        si, sj, gi, gj, xi, xj = state
        if xi >= 7 and xi - xj >= 5:
            return 1
        elif xj >= 7 and xj - xi >= 2:
            return 0
        elif (xi + xj) % 2 == 1:
            return self.Fij * self.P_tb((si, sj, gi + 1, gj, xi, xj)) + (
                1 - self.Fij
            ) * self.P_tb((si, sj, gi, gj + 1, xi, xj))
        else:
            return self.Fij * (1 - self.P_tb((si, sj, gj, gi + 1, xi, xj))) + (
                1 - self.Fij
            ) * self.P_tb((si, sj, gj + 1, gi, xi, xj))

    def P_m(self, state):
        if len(state) != 6:
            raise ValueError("state should be a tuple or a list")
        si, sj, gi, gj, xi, xj = state
        as_str = "".join(map(str, state))

        # Check in cache
        if as_str in self.P_m_cache:
            return self.P_m_cache[as_str]
        
        if si >= 3:
            p= 1
        elif sj >= 3:
            p =0
        else:
            p = self.P_s((si, sj, 0, 0, xi, xj)) * self.P_m(
                (si + 1, sj, gi, gj, xi, xj)
            ) + (1 - self.P_s((si, sj, 0, 0, xi, xj))) * self.P_m(
                (si, sj + 1, gi, gj, xi, xj)
            )
        # print(f"P_m({state}) = {p}")
        self.P_m_cache[state] = p
        return p

    def P_win(self, state):
        if len(state) != 6:
            raise ValueError("state should have 6 elements")

        si, sj, gi, gj, xi, xj = state

        as_str = "".join(map(str, state))

        # Check in cache
        if as_str in self.P_m_cache:
            return self.P_m_cache[as_str]
        
        p = self.Fij * self.P_m((si, sj, gi, gj, xi + 1, xj)) + (
            1 - self.Fij
        ) * self.P_m((si, sj, gi, gj, xi, xj + 1))

        global x
        x = x+1
        print(f"{x}.P_win({state}) {p}")
        
        # print("----------------",x)

        self.P_m_cache[as_str] = p
        return p
    

def plot_column_vs_columns(match_df, col, cols):
    fig, ax = plt.subplots(len(cols) // 2 + 2, 2, figsize=(20, 10))
    j = 0
    for i, column in enumerate(cols):
        sns.countplot(x=column, hue=col, data=match_df, ax=ax[j, i % 2])
        plt.title(f"{column} vs. {col}")
        plt.xlabel(column)
        plt.ylabel("Count")
        fig.legend(title="Point Victor")
        if i % 2 == 1:
            j += 1
    plt.savefig("Data/Plots/plot_column_vs_columns.png")


def plot_match_flow(tmp_match_df, coeff=None):
    # for i in [1, 2]:
    #     prob_set_diff, prob_game_diff = (0.44796814936847884, 0.39621087314662273)
    #     match_df.loc[:,f"p{i}_flow"] = (match_df[f"p{i}_sets"] - match_df[f"p{3-i}_sets"]) * prob_set_diff/2 + (match_df[f"p{i}_games"] - match_df[f"p{3-i}_games"]) *prob_game_diff/2 + (match_df[f"p{i}_points_won"] - match_df[f"p{3-i}_points_won"]) * 0.033

    map_score = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}
    # for i in [1, 2]:
    #     c1, c2, c3, c4, c5 = coeff
    #     match_df.loc[:, f"p{i}_flow"] = (
    #         (c1 / 2) * (match_df[f"p{i}_sets"] - match_df[f"p{3-i}_sets"])
    #         + (c2 / 5) * (match_df[f"p{i}_games"] - match_df[f"p{3-i}_games"])
    #         + c3 * (match_df["server"].apply(lambda x: 1 if x == i else 0))
    #         + (c4 / 3)
    #         * (
    #             match_df[f"p{i}_score"].apply(lambda x: map_score.get(x) if x in map_score else int(x))
    #             - match_df[f"p{3-i}_score"].apply(lambda x: map_score.get(x) if x in map_score else int(x))
    #         )
    #         + c5 * (match_df[f"p{i}_ace"].cumsum() - match_df[f"p{3-i}_ace"].cumsum())
    #     )

    prob = Probablities(0.66)
    
    match_df = tmp_match_df.copy()
    # if any row has p1_score as "AD" then replace it with 3 and p2_score with 2
    match_df.loc[:, "p1_score"] = match_df["p1_score"].apply(
        lambda x: map_score.get(x) if x in map_score else int(x)
    )
    match_df.loc[:, "p2_score"] = match_df["p2_score"].apply(
        lambda x: map_score.get(x) if x in map_score else int(x)
    )
    match_df[["p1_score", "p2_score"]] = match_df[["p1_score", "p2_score"]].where(
        match_df["p1_score"] != 4, [3, 2]
    )
    match_df[["p1_score", "p2_score"]] = match_df[["p1_score", "p2_score"]].where(
        match_df["p2_score"] != 4, [2, 3]
    )
    
    # print(match_df[["p1_sets","p2_sets","p1_games","p2_games","p1_score", "p2_score"]])
    # match_df.apply(lambda x: print(x["p1_sets"], x["p2_sets"], x["p1_games"], x["p2_games"], x["p1_score"], x["p2_score"]) if x["p1"], axis=1)

    # save the match_df to a csv file
    match_df.to_csv("Data/match_df.csv")
 
    # print("======================================")
    match_df.loc[:, f"p1_flow"] = match_df.apply(
        lambda x: prob.P_win(
            [
                int(x[f"p1_sets"]),
                int(x[f"p2_sets"]),
                int(x[f"p1_games"]),
                int(x[f"p2_games"]),
                int(x[f"p1_score"]),
                int(x[f"p2_score"]),
            ]
        ),
        axis=1,
    )

    # figure size
    plt.figure(figsize=(20, 10))
    sns.lineplot(x=match_df.index, y="p1_flow", data=match_df, linewidth=0.5)
    # Plot y = 0 line
    plt.axhline(0, color="black", linewidth=1.5, linestyle="-")

    # get the indexes of the set changes
    sets = divide_by_sets(match_df)
    for i, set in enumerate(sets):
        # draw vertical lines at the beginning of each set
        plt.axvline(
            set.index[-1],
            color="black",
            linewidth=0.5,
            linestyle="-",
            label=f"Set {i+1}",
        )
        plt.text(
            set.index[-1],
            0,
            f"Set {i+1}",
            rotation=90,
            verticalalignment="top",
            horizontalalignment="right",
            fontdict={"color": "black", "size": 8},
        )

    # fill with color blue if p1_flow > 0 and red if p1_flow < 0
    plt.fill_between(
        match_df.index,
        match_df["p1_flow"],
        0,
        where=match_df["p1_flow"] > 0,
        interpolate=True,
        color="blue",
        alpha=0.3,
    )
    plt.fill_between(
        match_df.index,
        match_df["p1_flow"],
        0,
        where=match_df["p1_flow"] < 0,
        interpolate=True,
        color="red",
        alpha=0.3,
    )
    # sns.lineplot(x=match_df.index, y="p2_flow", data=match_df, label="Player 2")
    plt.title("Match Flow")
    plt.xlabel("Point")
    plt.ylabel("Flow")
    plt.xticks(np.arange(match_df.index[0], match_df.index[-1], 15), rotation=90)
    # plt.legend(title='Players')
    plt.show()

    # sns.lineplot(x=match_df.index, y="p1_flow", data=match_df, label="Player 1")
    # sns.lineplot(x=match_df.index, y="p2_flow", data=match_df, label="Player 2")
    # plt.title("Match Flow")
    # plt.xlabel("Point")
    # plt.ylabel("Flow")
    # plt.xticks(np.arange(match_df.index[0], match_df.index[-1], 15), rotation=90)
    # plt.legend(title="Players")
    # plt.show()