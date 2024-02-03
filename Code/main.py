import csv
from read import *



rows = read_csv('./Data/Wimbledon_featured_matches.csv')
# rows = read_csv('./Data/game.csv')

def prob_point_given_server(rows):
    points_won = 0
    points_served = 0

    for row in rows:
        points_served += 1
        if row[13] == row[15]:
            points_won += 1

    prob_point_given_server = points_won/points_served
    print(f'Given a player serves, their probability of winning the point is {prob_point_given_server: .4%}')

def win_game_given_server(rows):
    games_won = 0
    games_served = 0

    for row in rows:
        if row[18] != '0':
            games_served += 1
            if row[13] == row[18]:
                games_won += 1

    if games_served == 0:
        print("No games played")
    else:
        win_game_given_server = games_won/games_served
        print(f'Given a player serves, their probability of winning the game is {win_game_given_server: .4%}')

def win_point_given_1fault(rows):
    points_won = 0
    points_served = 0

    for row in rows:
        if row[14] == '2':
            points_served += 1
            if row[13] == row[15]:
                points_won += 1

    if points_served == 0:
        print("No points served")
    else:
        win_point_given_1fault = points_won/points_served
        print(f'Given a player faults once, their probability of winning the point is {win_point_given_1fault: .4%}')

def win_point_given_2faultlast(rows):
    points_won_after = 0
    doubleFaults = 0
    faulter = 0
    prevMatch = ""

    for row in rows:
        currMatch = row[0]
        if not currMatch ==prevMatch:
            prevMatch = currMatch
            faulter = 0

        if faulter != 0:
            if int(row[15]) == faulter:
                points_won_after += 1
            faulter = 0
        else:
            if row[25] == '1':
                faulter = 1
                doubleFaults += 1
            if row[26] == '1':
                faulter = 2
                doubleFaults += 1
    if doubleFaults == 0:
        print("No Double Faults")
    else:
        win_point_given_2faultlast = points_won_after/doubleFaults
        print(f'Given a player double-faulted, their probability of winning the next point is {win_point_given_2faultlast: .4%}')



prob_point_given_server(rows)
win_game_given_server(rows)
win_point_given_1fault(rows)
win_point_given_2faultlast(rows)