import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import matplotlib.patches as mpatches

logging.basicConfig(filename="prob.log", level=logging.DEBUG)

df = pd.read_csv("Data/Wimbledon_featured_matches.csv")


def divide_by_matches(df):
    return [df[df["match_id"] == i] for i in df["match_id"].unique()]


def divide_by_sets(match_df):
    return [match_df[match_df["set_no"] == i] for i in match_df["set_no"].unique()]


def divide_by_games(set_df):
    return [set_df[set_df["game_no"] == i] for i in set_df["game_no"].unique()]


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
        match.loc[:, "match_victor"] = winner
    return pd.concat(match_df)


matches_df = divide_by_matches(df)


class MatchFlowModel:
    def __init__(self, Fij):
        self.Fij = Fij
        self.P_tb1_cache = {}
        self.P_tb2_cache = {}
        self.P_m_cache = {}

    def P_tb(self, state):  # Ambiguous
        si, sj, gi, gj, xi, xj = state

        if state in self.P_tb1_cache:
            return self.P_tb1_cache[state]

        if xi == xj and xj >= 7:
            p = 1 / 2

        elif xi >= 7 and xi - xj >= 2:
            p = self.P_m((si + 1, sj, 0, 0, 0, 0))

        elif xj >= 7 and xj - xi >= 2:
            p = self.P_m((si, sj + 1, 0, 0, 0, 0))

        elif (xi + xj) % 2 == 1:
            if xi > 9 or xj > 9:
                xi -= 2
                xj -= 2

            p = self.Fij * self.P_m((si, sj, gi, gj, xi + 1, xj)) + (
                1 - self.Fij
            ) * self.P_m((si, sj, gi, gj, xi, xj + 1))
        else:
            if xi > 9 or xj > 9:
                xi -= 2
                xj -= 2
            p = self.Fij * (1 - self.P_m((sj, si, gj, gi, xj, xi + 1))) + (
                1 - self.Fij
            ) * (1 - self.P_m((sj, si, gj, gi, xj + 1, xi)))

        self.P_tb1_cache[state] = p
        return p

    def P_tb5(self, state):
        si, sj, gi, gj, xi, xj = state

        if state in self.P_tb2_cache:
            return self.P_tb2_cache[state]

        if xi >= 10 and xi - xj >= 2:
            p = 1
        elif xj >= 10 and xj - xi >= 2:
            p = 0
        elif xi == 10 and xj == 10:
            p = 1 / 2
        elif (xi + xj) % 2 == 1:
            p = self.Fij * self.P_tb5((si, sj, gi, gj, xi + 1, xj)) + (
                1 - self.Fij
            ) * self.P_tb5((si, sj, gi, gj, xi, xj + 1))
        else:
            p = self.Fij * (1 - self.P_tb5((sj, si, gj, gi, xj, xi + 1))) + (
                1 - self.Fij
            ) * (1 - self.P_tb5((sj, si, gj, gi, xj + 1, xi)))

        self.P_tb2_cache[state] = p
        return p

    def P_m(self, state):
        si, sj, gi, gj, xi, xj = state
        # Check in cache
        if state in self.P_m_cache:
            return self.P_m_cache[state]

        if si >= 3:  # 1
            p = 1
        elif sj >= 3:  # 2
            p = 0
        elif (si == 2 and sj == 2) and (gi == 6 and gj == 6):  # 3
            p = self.P_tb5(state)
        elif (si != 2 or sj != 2) and (gi == 6 and gj == 6):  # 4
            p = self.P_tb(state)
        elif gi >= 6 and (gi - gj) >= 2:  # 5
            p = 1 - self.P_m((sj, si + 1, 0, 0, 0, 0))
        elif gj >= 6 and (gj - gi) >= 2:  # 6
            p = 1 - self.P_m((sj + 1, si, 0, 0, 0, 0))
        elif xi == 4 and xj <= 2:  # 7
            p = 1 - self.P_m((sj, si, gj, gi + 1, 0, 0))
        elif xj == 4 and xi <= 2:  # 8
            p = 1 - self.P_m((sj, si, gj + 1, gi, 0, 0))
        elif xi == 3 and xj == 3:
            p = ((self.Fij**2) / (2 * self.Fij**2 - 2 * self.Fij + 1)) * (
                1 - self.P_m((sj, si, gj, gi + 1, 0, 0))
            )
        else:
            p = self.Fij * self.P_m((si, sj, gi, gj, xi + 1, xj)) + (
                1 - self.Fij
            ) * self.P_m((si, sj, gi, gj, xi, xj + 1))

        self.P_m_cache[state] = p
        return p

    def P_win(self, state):
        return self.P_m(state)


def plot_match_flow(tmp_match_df):
    map_score = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4}

    prob = MatchFlowModel(0.66)

    match_df = tmp_match_df.copy()

    # Data Cleaning to account for "AD"
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

    match_df.loc[:, f"p1_flow"] = match_df.apply(
        lambda x: prob.P_win(
            (
                int(x[f"p1_sets"]),
                int(x[f"p2_sets"]),
                int(x[f"p1_games"]),
                int(x[f"p2_games"]),
                int(x[f"p1_score"]),
                int(x[f"p2_score"]),
            )
        ),
        axis=1,
    )

    plt.figure(figsize=(25, 10))
    sns.lineplot(
        x=(match_df.index - match_df.index[0]),
        y="p1_flow",
        data=match_df,
        linewidth=0.5,
        color="black",
    )
    plt.axhline(0.5, color="black", linewidth=1.5, linestyle="-")

    sets = divide_by_sets(match_df)
    for i, set in enumerate(sets):
        center = (
            (set.index[0] - match_df.index[0]) + (set.index[-1] - match_df.index[0])
        ) / 2
        plt.axvline(
            set.index[-1] - match_df.index[0],
            color="black",
            linewidth=0.5,
            linestyle="-",
            label=f"Set {i+1}",
        )
        plt.text(
            center,
            0.85,  # max value in the axis
            f"Set {i+1}",
            backgroundcolor="white",
            rotation=0,
            verticalalignment="top",
            horizontalalignment="right",
            fontdict={"color": "black", "size": 12},
        )

    plt.fill_between(
        match_df.index - match_df.index[0],
        match_df["p1_flow"],
        0.5,
        where=match_df["p1_flow"] > 0.5,
        interpolate=True,
        color="blue",
        alpha=0.3,
    )
    plt.fill_between(
        match_df.index - match_df.index[0],
        match_df["p1_flow"],
        0.5,
        where=match_df["p1_flow"] < 0.5,
        interpolate=True,
        color="red",
        alpha=0.3,
    )
    plt.title("Match Flow for 2023 Wimbledon Gentlemen's Final", fontdict={"size": 20})
    plt.xlabel("Point", fontdict={"size": 15})
    plt.ylabel("Probability of Winning the Match", fontdict={"size": 15})
    plt.xticks(np.arange(0, match_df.index[-1] - match_df.index[0], 15), rotation=0)
    i0 = match_df.index[0]
    plt.legend(
        title="Players",
        handles=[
            mpatches.Patch(color="blue", alpha=0.3, label=match_df.at[i0, "player1"]),
            mpatches.Patch(color="red", alpha=0.3, label=match_df.at[i0, "player2"]),
        ],
        loc="upper left",
        fontsize=12,
    )

    # plt.savefig("Data/match_flow_final.png")
    plt.show()


# Champion Ship Match plot
match_number_in_dataset = -1
plot_match_flow(matches_df[match_number_in_dataset])
