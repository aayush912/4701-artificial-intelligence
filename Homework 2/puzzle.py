from __future__ import division
from __future__ import print_function
from os import stat

import sys
import math
import time
import queue as Q
import heapq
import resource


#### SKELETON CODE ####
## The Class that Represents the Puzzle
class PuzzleState(object):
    """
        The PuzzleState stores a board configuration and implements
        movement instructions to generate valid children.
    """
    def __init__(self, config, n, parent=None, action="Initial", cost=0):
        """
        :param config->List : Represents the n*n board, for e.g. [0,1,2,3,4,5,6,7,8] represents the goal state.
        :param n->int : Size of the board
        :param parent->PuzzleState
        :param action->string
        :param cost->int
        """
        if n*n != len(config) or n < 2:
            raise Exception("The length of config is not correct!")
        if set(config) != set(range(n*n)):
            raise Exception("Config contains invalid/duplicate entries : ", config)

        self.n        = n
        self.cost     = cost
        self.parent   = parent
        self.action   = action
        self.config   = config
        self.children = []

        # Get the index and (row, col) of empty block
        self.blank_index = self.config.index(0)

    def display(self):
        """ Display this Puzzle state as a n*n board """
        for i in range(self.n):
            print(self.config[3*i : 3*(i+1)])

    def move_up(self):
        """ 
        Moves the blank tile one row up.
        :return a PuzzleState with the new configuration
        """
        node = PuzzleState(list(self.config), self.n, parent= self, action='Up', cost=self.cost+1)
        for i in range(len(node.config)):
            if node.config[i] == 0:
                if i != 0 and i != 1 and i != 2:
                    node.config[i], node.config[i-3] = node.config[i-3], node.config[i]
                    return node
        return None
      
    def move_down(self):
        """
        Moves the blank tile one row down.
        :return a PuzzleState with the new configuration
        """
        node = PuzzleState(list(self.config), self.n, parent= self, action='Down', cost=self.cost+1)
        for i in range(len(node.config)):
            if node.config[i] == 0:
                if i != 6 and i != 7 and i != 8:
                    node.config[i], node.config[i+3] = node.config[i+3], node.config[i]
                    return node
        return None
        
      
    def move_left(self):
        """
        Moves the blank tile one column to the left.
        :return a PuzzleState with the new configuration
        """
        node = PuzzleState(list(self.config), self.n, parent= self, action='Left', cost=self.cost+1)
        for i in range(len(node.config)):
            if node.config[i] == 0:
                if i != 0 and i != 3 and i != 6:
                    node.config[i], node.config[i-1] = node.config[i-1], node.config[i]
                    return node
        return None
        

    def move_right(self):
        """
        Moves the blank tile one column to the right.
        :return a PuzzleState with the new configuration
        """
        node = PuzzleState(list(self.config), self.n, parent= self, action='Right', cost=self.cost+1)
        for i in range(len(node.config)):
            if node.config[i] == 0:
                if i != 2 and i != 5 and i != 8:
                    node.config[i], node.config[i+1] = node.config[i+1], node.config[i]
                    return node
        return None
        
      
    def expand(self):
        """ Generate the child nodes of this node """
        
        # Node has already been expanded
        if len(self.children) != 0:
            return self.children
        
        # Add child nodes in order of UDLR
        children = [
            self.move_up(),
            self.move_down(),
            self.move_left(),
            self.move_right()]

        # Compose self.children of all non-None children states
        self.children = [state for state in children if state is not None]
        return self.children

    def __eq__(self, other):
        return self.config == other.config

    def __hash__(self):
        return hash(tuple(self.config))
    
# Function that Writes to output.txt

### Students need to change the method to have the corresponding parameters
def writeOutput(final_state, depth, max_depth, expanded, ram, r_time):
    ### Student Code Goes here
    state = final_state
    path = []
    while state is not None:
        path.append (state.action)
        state = state.parent
    path = path [:-1]
    path = path [::-1]
    f = open("output.txt", "w")
    f.write("path_to_goal:" + str(path) + "\n")
    f.write("cost_to_goal:" + str(final_state.cost) + "\n")
    f.write("nodes_expanded:" + str(expanded) + "\n")
    f.write("search_depth:" + str(depth) + "\n")
    f.write("max_search_depth:" + str(max_depth) + "\n")
    f.write("running_time:" + str(round(r_time,8)) + "\n")
    f.write("max_ram_usage:" + str(round(ram,8)) + "\n")
    f.close()

def bfs_search(initial_state):
    """BFS search"""
    ### STUDENT CODE GOES HERE ###
    start = time.time()
    frontier = []
    frontier_set = set()
    explored = set()
    frontier.append(initial_state)
    frontier_set.add(initial_state)
    depth, expanded = 0, 0
    while (len(frontier) > 0):
        state = frontier.pop(0)
        frontier_set.remove(state)
        explored.add(state)
        if (test_goal(state) == True):
            end = time.time()
            writeOutput(state, state.cost, depth, expanded, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, end - start)
            return
        expanded = expanded + 1
        neighbors = state.expand()
        for neighbor in neighbors:
            if neighbor not in frontier_set and neighbor not in explored:
                frontier.append(neighbor)
                frontier_set.add(neighbor)
                depth = max(depth, neighbor.cost)

def dfs_search(initial_state):
    """DFS search"""
    ### STUDENT CODE GOES HERE ###
    start = time.time()
    frontier = []
    frontier_set = set()
    explored = set()
    depth, expanded = 0, 0
    frontier.append(initial_state)
    frontier_set.add(initial_state)
    while (len(frontier) > 0):
        state = frontier.pop(len(frontier) - 1)
        frontier_set.remove(state)
        explored.add(state)
        if (test_goal(state) == True):
            end = time.time()
            writeOutput(state, state.cost, depth, expanded, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, end - start)
            return
        expanded = expanded + 1
        neighbors = state.expand()
        for neighbor in reversed(neighbors):
            if neighbor not in frontier_set and neighbor not in explored:
                frontier.append(neighbor)
                frontier_set.add(neighbor)
                depth = max(depth, neighbor.cost)

def A_star_search(initial_state):
    """A * search"""
    start = time.time()
    index = 0
    explored = set()
    frontier_dict = {}
    frontier = []
    fn = calculate_total_cost(initial_state)
    comparatorList = [fn, index, initial_state]
    heapq.heappush(frontier, comparatorList)
    index = index + 1
    frontier_dict[initial_state] = fn
    depth, expanded = 0, 0
    while len(frontier) > 0:
        state = heapq.heappop(frontier)[2]
        frontier_dict.pop(state)
        explored.add(state)
        if test_goal(state) == True:
            end = time.time()
            writeOutput(state, state.cost, depth, expanded, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, end - start)
            return
        expanded = expanded + 1
        neighbors = state.expand()
        for neighbor in neighbors:
            est_cost = calculate_total_cost(neighbor)
            if neighbor not in frontier_dict and neighbor not in explored:
                depth = max(depth, neighbor.cost)
                entry = [est_cost, index, neighbor]
                heapq.heappush(frontier, entry)
                index = index + 1
                frontier_dict[neighbor] = est_cost
            elif neighbor in frontier_dict:
                if frontier_dict[neighbor] > est_cost:
                    frontier_dict[neighbor] = est_cost
                    neighbor.parent = state
                    neighbor.cost = state.cost + 1
                    for key, value in enumerate(frontier):
                        if neighbor == value[2]:
                            frontier.pop(key)
                    heapq.heappush(frontier, [est_cost, index, neighbor])
                    index = index + 1                    


def calculate_total_cost(state):
    """calculate the total estimated cost of a state"""
    ### STUDENT CODE GOES HERE ###
    size = len(state.config)
    dis = 0
    for i in range (1, size):
        idx = state.config.index(i)
        dis = dis + calculate_manhattan_dist(idx, i, state.n)
    return (state.cost + dis)

def calculate_manhattan_dist(idx, value, n):
    """calculate the manhattan distance of a tile"""
    ### STUDENT CODE GOES HERE ###
    if value > 0:
        xg, yg = value // n, value % n
        xa, ya = idx // n, idx % n
    return abs (xg - xa) + abs (yg - ya)

def test_goal(puzzle_state):
    """test the state is the goal state or not"""
    for i in range (len(puzzle_state.config)):
        if puzzle_state.config[i] != i:
            return False
    return True

    

# Main Function that reads in Input and Runs corresponding Algorithm
def main():
    search_mode = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = list(map(int, begin_state))
    board_size  = int(math.sqrt(len(begin_state)))
    hard_state  = PuzzleState(begin_state, board_size)
    start_time  = time.time()
    
    if   search_mode == "bfs": bfs_search(hard_state)
    elif search_mode == "dfs": dfs_search(hard_state)
    elif search_mode == "ast": A_star_search(hard_state)
    else: 
        print("Enter valid command arguments !")
        
    end_time = time.time()
    print("Program completed in %.3f second(s)"%(end_time-start_time))

if __name__ == '__main__':
    main()
