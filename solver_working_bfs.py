#!/usr/bin/python
import sys
from laser_tank import LaserTankMap

"""
Template file for you to implement your solution to Assignment 1.

COMP3702 2020 Assignment 1 Support Code
"""
import csv


from collections import deque
from operator import sub
from operator import add


def find_neighbours(lmap, input_cell):
    
    grid_data = lmap.grid_data
    connections =  [[0,1], [0,-1], [1,0], [-1,0]]
    neighbours = [find_next_cell(input_cell, connection) for connection in connections]
    neighbours = [cell for cell in neighbours if not lmap.cell_is_blocked(cell[0], cell[1])]
    neighbours = [cell for cell in neighbours if not lmap.cell_is_game_over(cell[0], cell[1])]
    
    return neighbours



def find_next_cell(cell, connection):
        
    from operator import add
    from operator import sub
    
    return list( map(add, cell, connection) )


def passable(grid_data, cell):
        ''' 
        check if node is not blocks
        '''
        list1 = find_blockage(grid_data)
        return (cell not in list1)
    
def find_blockage(grid_data):
    blockage_list = []
    for i, row in enumerate(grid_data):
        for j, cell in enumerate(row):
            if cell in [obs,wat]:
                blockage_list.append([i,j])
    
    return blockage_list
            
    
def find_flag(lmap):
    flag = lmap.FLAG_SYMBOL
    grid = lmap.grid_data
    for i,row in enumerate(grid):
        for j,block in enumerate(row):
            if block == flag:
                x = j
                y = i
                return(list([y,x]))

    

def BFS(lmap, start, goal):
    #x = lmap.player_x
    #y = lmap.player_y
    #
    #start = [y,x]
    #goal = find_flag(lmap)
    
    
    frontier = deque() 
    frontier.append(start)
    visited = []
    visited.append(start)
    path = {}
    backtrack = {}
    backtrack[tuple(start)] = tuple(start)
    
    
    
    path[tuple(start)] = [0,0]
    
    while len(frontier) > 0:
        current = frontier.popleft()
        #print('current node = ', current)
        #print('goal node = ', goal)
        
        if current == tuple(goal):
            break
        neighbours = find_neighbours(lmap,current)
        
        
        
        for next in neighbours:
           #if next not in visited:
           #    frontier.append(next)
           #    visited.append(next)
           #
            
            if tuple(next) not in path:
                frontier.append(tuple(next))
                backtrack[tuple(next)] = tuple(current)
                path[tuple(next)] = list( map(sub, current, next))
    
    #getting path from start to goal
    
    #current = list( map(add, start, path[tuple(start)]))
    #while current != goal:
    #    print(current)
    #    current = list( map(add, current, path[tuple(current)]))
    #    
        
    
    #print('total nodes searched = ', len(path))
    #print(path)
    
    if tuple(goal) not in backtrack:
        #print ('dead end reach. no path found.')
        return
    
    path_list = [goal,start]
    
    key = tuple(goal)
    
    
    while backtrack[key] != tuple(start):
        #print(backtrack[key])
        path_list.insert(-1,list(backtrack[key]))
        key = backtrack[key]
        
    
    path_list.reverse()
    
    return path_list   


def get_moves(lmap, start, end):
    
    path = BFS(lmap, start, end)
    
    if path is None:
        print('No path found.')
        return
    MOVE_FORWARD = 'f'
    TURN_LEFT = 'l'
    TURN_RIGHT = 'r'
    SHOOT_LASER = 's'
    MOVES = [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, SHOOT_LASER]
    # directions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # move return statuses
    SUCCESS = 0
    COLLISION = 1
    GAME_OVER = 2
    moves_list = []
    
    player_heading = lmap.player_heading
    
    for i in range(len(path) - 1):
    
        move = list( map(sub, path[i], path[i+1]))
        if move == [0, 1]:
            
            if player_heading == UP :
                #print('left')
                moves_list.append('l')
                player_heading = 2
                #print('forward')
                moves_list.append('f')
            elif player_heading == DOWN :
                #print('right')
                moves_list.append('r')
                player_heading = 2
                
                #print('forward')
                moves_list.append('f')
            elif player_heading == LEFT :
                player_heading = 2
                
                #print('forward')
                moves_list.append('f')
            elif player_heading == RIGHT :
                #print('left')
                moves_list.append('l')
                player_heading = 2
                #print('left')
                moves_list.append('l')
                #print('forward')
                moves_list.append('f')
            
            
        elif move == [0, -1]:
            
            if player_heading == UP :
                #print('right')
                moves_list.append('r')
                player_heading = 3
                #print('forward')
                moves_list.append('f')
            elif player_heading == DOWN :
                #print('left')
                moves_list.append('l')
                player_heading = 3
                #print('forward')
                moves_list.append('f')
            elif player_heading == LEFT :
                #print('left')
                moves_list.append('l')
                #print('left')
                moves_list.append('l')
                player_heading = 3
                #print('forward')
                moves_list.append('f')
            elif player_heading == RIGHT :
                #print('forward')
                moves_list.append('f')
            
            
            
        elif move == [1, 0]:
            
            if player_heading == UP :
                #print('forward')
                moves_list.append('f')
            elif player_heading == DOWN :
                #print('left')
                moves_list.append('l')
                #print('left')
                moves_list.append('l')
                player_heading = 0
                #print('forward')
                moves_list.append('f')
            elif player_heading == LEFT :
                #print('right')
                moves_list.append('r')
                player_heading = 0
                #print('forward')
                moves_list.append('f')
            elif player_heading == RIGHT :
                #print('left')
                moves_list.append('l')
                player_heading = 0
                #print('forward')
                moves_list.append('f')
            
            
        elif move == [-1,0]:
            
            if player_heading == UP :
                #print('left')
                moves_list.append('l')
                #print('left')
                moves_list.append('l')
                player_heading = 1
                #print('forward')
                moves_list.append('f')
            elif player_heading == DOWN :
                #print('forward')
                moves_list.append('f')
            elif player_heading == LEFT :
                #print('left')
                moves_list.append('l')
                player_heading = 1
                #print('forward')
                moves_list.append('f')
            elif player_heading == RIGHT :
                #print('right')
                moves_list.append('r')
                player_heading = 1
                #print('forward')
                moves_list.append('f')
    return moves_list
    
        
        
  

#
#
# Code for any classes or functions you need can go here.
#
#




def write_output_file(filename, actions):
    """
    Write a list of actions to an output file. You should use this method to write your output file.
    :param filename: name of output file
    :param actions: list of actions where is action is in LaserTankMap.MOVES
    """
    f = open(filename, 'w')
    for i in range(len(actions)):
        f.write(str(actions[i]))
        if i < len(actions) - 1:
            f.write(',')
    f.write('\n')
    f.close()


def main(arglist):
    
    
    
    input_file = arglist[0]
    output_file = arglist[1]
    

    # Read the input testcase file
    game_map = LaserTankMap.process_input_file(input_file)
    
    start = [game_map.player_y, game_map.player_x]
    end = find_flag(game_map)
    

    actions = get_moves(game_map,start,end)

    #
    #
    # Code for your main method can go here.
    #
    # Your code should find a sequence of actions for the agent to follow to reach the goal, and store this sequence
    # in 'actions'.
    #
    #

    # Write the solution to the output file
    write_output_file(output_file, actions)


if __name__ == '__main__':
    main(sys.argv[1:])
    


