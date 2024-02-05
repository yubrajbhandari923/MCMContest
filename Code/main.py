import csv
import statistics
from read import *



rows = read_csv('./Data/Wimbledon_featured_matches.csv')
# rows = read_csv('./Data/match.csv')

def prob_point_given_server(rows):
    points_won = 0
    points_served = 0

    for row in rows:
        points_served += 1
        if row[13] == row[15]:
            points_won += 1

    prob_point_given_server = points_won/points_served
    print(f'Given a player serves, their probability of winning the point is: {prob_point_given_server:.4%}')

def prob_point_given_returner(rows):
    points_won = 0
    points_returned = 0

    for row in rows:
        points_returned += 1
        if row[13] != row[15]:
            points_won += 1

    prob_point_given_returner = points_won/points_returned
    print(f'Given a player returns, their probability of winning the point is: {prob_point_given_returner:.4%}')

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

def proportions(rows):
    aces1 = 0
    aces2 = 0
    aceProp = 0
    win1 = 0
    win2 = 0
    winProp = 0
    fault1 = 0
    fault2 = 0
    fProp = 0
    fMatches = 31
    err1 = 0
    err2 = 0
    errProp = 0
    run1 = 0
    run2 = 0
    runProp = 0

    for row in rows:
        run1 += float(row[39])
        run2 += float(row[40])
        if row[20] == '1':
            aces1 += 1
        if row[21] == '1':
            aces2 += 1
        if row[22] == '1':
            win1 += 1
        if row[23] == '1':
            win2 += 1
        if row[25] == '1':
            fault1 += 1
        if row[26] == '1':
            fault2 += 1
        if row[27] == '1':
            err1 += 1
        if row[28] == '1':
            err2 += 1
        if row[19] == '1' and row[7] == '2':
            aceProp += (aces1/aces2)
            winProp += (win1/win2)
            if (fault2 != 0):
                fProp += (fault1/fault2)
            else:
                fMatches -= 1
            errProp += (err1/err2)
            runProp += (run1/run2)
            aces1 = 0
            aces2 = 0
            win1 = 0
            win2 = 0
            fault1 = 0
            fault2 = 0
            err1 = 0
            err2 = 0
            run1 = 0
            run2 = 0
        if row[19] == '2' and row[8] == '2':
            aceProp += (aces2/aces1)
            winProp += (win2/win1)
            if (fault1 != 0):
                fProp += (fault2/fault1)
            else:
                fMatches -= 1
            errProp += (err2/err1)
            runProp += (run2/run1)
            aces1 = 0
            aces2 = 0
            win1 = 0
            win2 = 0
            fault1 = 0
            fault2 = 0
            err1 = 0
            err2 = 0
            run1 = 0
            run2 = 0
    
    prop1 = aceProp/31
    prop2 = winProp/31
    prop3 = fProp/fMatches
    prop4 = errProp/31
    prop5 = runProp/31
    print(f'Given a player wins the match, their proportion of aces versus that of their opponent is: {prop1}')
    print(f'Given a player wins the match, their proportion of winners versus that of their opponent is: {prop2}')
    print(f'Given a player wins the match, their proportion of double-faults versus that of their opponent is: {prop3}')
    print(f'Given a player wins the match, their proportion of unforced errors versus that of their opponent is: {prop4}')
    print(f'Given a player wins the match, their proportion of meters run versus that of their opponent is: {prop5}')

def flow(rows):
    for row in rows:
        p = 2/3

        s1 = int(row[7])
        s2 = int(row[8])
        g1 = int(row[9])
        g2 = int(row[10])

        if row[11] == 'AD':
            p1 = 3
            p2 = 2
        elif row[12] == 'AD':
            p1 = 2
            p2 = 3
        else: 
            if row[11] == '15':
                p1 = 1
            elif row[11] == '30':
                p1 = 2
            elif row[11] == '40':
                p1 = 3
            else:
                p1 = int(row[11])
            
            if row[12] == '15':
                p2 = 1
            elif row[12] == '30':
                p2 = 2
            elif row[12] == '40':
                p2 = 3
            else:
                p2 = int(row[12])
        
        
        
        Prob = [[[[[[0 for _ in range(6)]
                for _ in range(6)]
                for _ in range(6)]
                for _ in range(6)]
                for _ in range(6)]
                for _ in range(6)]
        


        Pm(si, sj , gi, gj , xi, xj ) = fij ∗ Pm(si, sj , gi, gj , xi + 1, xj ) + (1 − fij )Pm(si, sj , gi, gj , xi, xj + 1)




# err = 0
# rowNum = 0
# for row in rows:
#     rowNum += 1
#     if not (row[11] == '0' or row[11] == '15' or row[11] == '30' or row[11] == '40' or row[11] == 'AD'):
#         err += 1
#         print(rowNum)
#     if not (row[12] == '0' or row[12] == '15' or row[12] == '30' or row[12] == '40' or row[12] == 'AD'):
#         err += 1
#         print(rowNum)
# print(err)


# prob_point_given_server(rows)
# prob_point_given_returner(rows)
# win_game_given_server(rows)
# win_point_given_1fault(rows)
# win_point_given_2faultlast(rows)
# win_set_given_prevSet(rows)
# win_game_given_prevGame(rows)
# averages(rows)
# proportions(rows)


# print(f'mean: {statistics.mean()}')
# print(f'median: {statistics.median()}')
# print(f'standard deviation: {statistics.stdev()}')
# print(f'quartiles: {statistics.quantiles(rows, n=4)}')
# print(f'minimum: {min()}')
# print(f'maximum: {max()}')

# do for rally, distance, speed, and try serve width, depth, and return depth