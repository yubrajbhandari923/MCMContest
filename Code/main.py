import csv
import statistics
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
    print(f'Given a player serves, their probability of winning the point is: {prob_point_given_server:.4%}')

def win_game_given_server(rows):
    games_won = 0
    games_served = 0

    for row in rows:
        if row[18] != '0':
            games_served += 1
            if row[13] == row[18]:
                games_won += 1

    if games_served == 0:
        print("No Data")
    else:
        win_game_given_server = games_won/games_served
        print(f'Given a player serves, their probability of winning the game is: {win_game_given_server:.4%}')

def win_point_given_1fault(rows):
    points_won = 0
    points_served = 0

    for row in rows:
        if row[14] == '2':
            points_served += 1
            if row[13] == row[15]:
                points_won += 1

    if points_served == 0:
        print("No Data")
    else:
        win_point_given_1fault = points_won/points_served
        print(f'Given a player faults once, their probability of winning the point is: {win_point_given_1fault:.4%}')

def win_point_given_2faultlast(rows):
    points_won_after = 0
    doubleFaults = 0
    faulter = 0
    prevMatch = ""

    for row in rows:
        currMatch = row[0]
        if not currMatch == prevMatch:
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
        print("No Data")
    else:
        win_point_given_2faultlast = points_won_after/doubleFaults
        print(f'Given a player double-faulted, their probability of winning the next point is: {win_point_given_2faultlast:.4%}')

def win_set_given_prevSet(rows):
    next_won = 0
    sets_won = 0
    winner = 0
    prevMatch = ""
    matches = 0

    for row in rows:
        currMatch = row[0]
        if not currMatch == prevMatch:
            prevMatch = currMatch
            winner = 0
            matches += 1

        if row[19] != '0':
            sets_won += 1
            if int(row[19]) == winner:
                next_won += 1
            winner = int(row[19])

    if sets_won == 0:
        print("No Data")
    else:
        win_set_given_prevSet = next_won/(sets_won - matches)
        print(f'Given a player won their previous set, their probability of winning the next set is: {win_set_given_prevSet:.4%}')

def win_game_given_prevGame(rows):
    next_won = 0
    games_won = 0
    winner = 0
    prevMatch = ""
    matches = 0

    for row in rows:
        currMatch = row[0]
        if not currMatch == prevMatch:
            prevMatch = currMatch
            winner = 0
            matches += 1

        if row[18] != '0':
            games_won += 1
            if int(row[18]) == winner:
                next_won += 1
            winner = int(row[18])

    if games_won == 0:
        print("No Data")
    else:
        win_game_given_prevGame = next_won/(games_won - matches)
        print(f'Given a player won their previous game, their probability of winning the next game is: {win_game_given_prevGame:.4%}')

def averages(rows):
    totRally = 0
    totDist = 0
    totSpeed = 0
    naCount = 0

    for row in rows:
        totRally += int(row[41])
        totDist += float(row[39]) + float(row[40])
        if row[42] == 'NA':
            naCount += 1
            continue
        else:
            totSpeed += float(row[42])

    aveRally = totRally/len(rows)
    aveDist = totDist/(2*len(rows))
    aveSpeed = totSpeed/(len(rows) - naCount)
    print(f'Average Rally Per Point is: {aveRally}')
    print(f'Average Distance Ran Per Point is: {aveDist}')
    print(f'Average Speed Per Serve is: {aveSpeed}')


# prob_point_given_server(rows)
# win_game_given_server(rows)
# win_point_given_1fault(rows)
# win_point_given_2faultlast(rows)
# win_set_given_prevSet(rows)
# win_game_given_prevGame(rows)
averages(rows)



# print(f'mean: {statistics.mean()}')
# print(f'median: {statistics.median()}')
# print(f'standard deviation: {statistics.stdev()}')
# print(f'quartiles: {statistics.quantiles(rows, n=4)}')
# print(f'minimum: {min()}')
# print(f'maximum: {max()}')

# do for rally, distance, speed, and try serve width, depth, and return depth