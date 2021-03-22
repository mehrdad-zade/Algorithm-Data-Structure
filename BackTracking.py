'''
Backtracking is an algorithmic paradigm that tries different solutions until finds a 
solution that “works”. Problems which are typically solved using backtracking technique 
have the following property in common. These problems can only be solved by trying every 
possible configuration and each configuration is tried only once. A Naive solution for 
these problems is to try all configurations and output a configuration that follows given 
problem constraints. Backtracking works in an incremental way and is an optimization over 
the Naive solution where all possible configurations are generated and tried.

'''

#1############################################################################################


'''
Write a program to print all permutations of a given string

Below are the permutations of string ABC. 
ABC ACB BAC BCA CBA CAB

solution: https://media.geeksforgeeks.org/wp-content/cdn-uploads/NewPermutation.gif

'''

def toString(l):
    return ''.join(l)
    
def permute(a, l, r):
    if l==r:
        print(toString(a))
    else:
        for i in range(l, r+1):
            a[i], a[l] = a[l], a[i]
            permute(a, l+1, r)
            a[i], a[l] = a[l], a[i] # backtrack
            
string = "ABCD"
n = len(string) 
a = list(string) 
permute(a, 0, n-1)   

#2############################################################################################
'''
The Knight’s tour problem

Knight (horse) has 8 possible moves. given a square board, find out the incremental positions
that the knight should travel to, so that it can visit the entire board.

print the board as the result, showing all the steps the knight had traveled.
'''

n = 8

def isSafe(x, y, board):
    if x>=0 and x<n and y>=0 and y<n and board[x][y]==-1:
        return True
    return False
    
def printBoard(board):
    for i in range(n):
        for j in range(n):
            print(board[i][j], end=' ')
        print()
        
def knightTour():
    board = [[-1 for i in range(n)] for j in range(n)] #initialize board with -1
    
    position, curr_x, curr_y = 1, 0, 0
    board[curr_x][curr_y] = 0
    
    x_move = [2, 1, -1, -2, -2, -1, 1, 2]
    y_move = [1, 2, 2, 1, -1, -2, -2, -1]
    
    if knightTour_util(board, x_move, y_move, curr_x, curr_y, position):
        printBoard(board)
    else:
        print("Not Possible..")
        
def knightTour_util(board, x_move, y_move, curr_x, curr_y, position):
    if position == n**2:
        return True
    
    for i in range(n):
        new_x = curr_x + x_move[i]
        new_y = curr_y + y_move[i]
        if isSafe(new_x, new_y, board):
            board[new_x][new_y] = position
            if knightTour_util(board, x_move, y_move, new_x, new_y, position+1) == True:
                return True
            board[new_x][new_y] = -1 # backtracking
    return False
    
knightTour()
    
    
    

#3############################################################################################
'''
Rat in a Maze

A Maze is given as N*N binary matrix of blocks where source block is the upper left most block i.e., 
maze[0][0] and destination block is lower rightmost block i.e., maze[N-1][N-1]. A rat starts from source 
and has to reach the destination. The rat can move only in two directions: forward and down. 

In the maze matrix, 0 means the block is a dead end and 1 means the block can be used in the path from 
source to destination. 
'''

def printSolution( sol ):
     
    for i in sol:
        for j in i:
            print(str(j) + " ", end ="")
        print("")
 
# A utility function to check if x, y is valid
# index for N * N Maze
def isSafe( maze, x, y ):
     
    if x >= 0 and x < N and y >= 0 and y < N and maze[x][y] == 1:
        return True
     
    return False

def solveMaze( maze ):
     
    # Creating a 4 * 4 
    sol = [ [ 0 for j in range(4) ] for i in range(4) ]
     
    if solveMazeUtil(maze, 0, 0, sol) == False:
        print("Solution doesn't exist");
        return False
     
    printSolution(sol)
    return True
     
# A recursive utility function to solve Maze problem
def solveMazeUtil(maze, x, y, sol):
     
    # if (x, y is goal) return True
    if x == N - 1 and y == N - 1 and maze[x][y]== 1:
        sol[x][y] = 1
        return True
         
    # Check if maze[x][y] is valid
    if isSafe(maze, x, y) == True:
        # Check if the current block is already part of solution path.    
        if sol[x][y] == 1:
            return False
           
        # mark x, y as part of solution path
        sol[x][y] = 1
         
        # Move forward in x direction
        if solveMazeUtil(maze, x + 1, y, sol) == True:
            return True
             
        # If moving in x direction doesn't give solution 
        # then Move down in y direction
        if solveMazeUtil(maze, x, y + 1, sol) == True:
            return True
           
        # If moving in y direction doesn't give solution then 
        # Move back in x direction
        if solveMazeUtil(maze, x - 1, y, sol) == True:
            return True
             
        # If moving in backwards in x direction doesn't give solution 
        # then Move upwards in y direction
        if solveMazeUtil(maze, x, y - 1, sol) == True:
            return True
         
        # If none of the above movements work then 
        # BACKTRACK: unmark x, y as part of solution path
        sol[x][y] = 0
        return False
 
# Driver program to test above function
if __name__ == "__main__":
    maze = [ [1, 0, 0, 0],
             [1, 1, 0, 1],
             [0, 1, 0, 0],
             [1, 1, 1, 1] ]
              
    solveMaze(maze)

#4############################################################################################
'''
N Queen Problem
'''



global N 
N = 4
  
def printSolution(board): 
    for i in range(N): 
        for j in range(N): 
            print (board[i][j], end = " ") 
        print() 
  
# A utility function to check if a queen can 
# be placed on board[row][col]. Note that this 
# function is called when "col" queens are 
# already placed in columns from 0 to col -1. 
# So we need to check only left side for 
# attacking queens 
def isSafe(board, row, col): 
  
    # Check this row on left side 
    for i in range(col): 
        if board[row][i] == 1: 
            return False
  
    # Check upper diagonal on left side 
    for i, j in zip(range(row, -1, -1),  
                    range(col, -1, -1)): 
        if board[i][j] == 1: 
            return False
  
    # Check lower diagonal on left side 
    for i, j in zip(range(row, N, 1),  
                    range(col, -1, -1)): 
        if board[i][j] == 1: 
            return False
  
    return True
    
 
def solveNQ(): 
    board = [ [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0], 
              [0, 0, 0, 0] ] 
  
    if solveNQUtil(board, 0) == False: 
        print ("Solution does not exist") 
        return False
  
    printSolution(board) 
    return True
  
def solveNQUtil(board, col): 
      
    # base case: If all queens are placed 
    # then return true 
    if col >= N: 
        return True
  
    # try placing this queen in all rows one by one 
    for i in range(N): 
        
        if isSafe(board, i, col): 
              
            # Place this queen in board[i][col] 
            board[i][col] = 1
  
            # recur to place rest of the queens 
            if solveNQUtil(board, col + 1) == True: 
                return True
  
            # If placing queen in board[i][col 
            # doesn't lead to a solution, then BACKTRACK
            # queen from board[i][col] 
            board[i][col] = 0
  
    # if the queen can not be placed in any row in 
    # this colum col then return false 
    return False
 
  
# Driver Code 
solveNQ() 


#5############################################################################################
'''
subset sum

Subset sum problem is to find subset of elements that are selected from a given set whose sum adds up to a given number K.

Sol:
see the tree:
https://www.geeksforgeeks.org/subset-sum-backtracking-4/

In the above tree, a node represents function call and a branch represents candidate element. The root node contains 4 children. 
In other words, root considers every element of the set as different branch. The next level sub-trees correspond to the subsets 
that includes the parent node. The branches at each level represent tuple element to be considered. For example, if we are at 
level 1, tuple_vector[1] can take any value of four branches generated. If we are at level 2 of left most node, tuple_vector[2] 
can take any value of three branches generated, and so on…

As we go down along depth of tree we add elements so far, and if the added sum is satisfying explicit constraints, we will continue 
to generate child nodes further. Whenever the constraints are not met, we stop further generation of sub-trees of that node, and 
backtrack to previous node to explore the nodes not yet explored. In many scenarios, it saves considerable amount of processing time.

below solution is simple recursion and not using backtracking
'''

def isSubsetSum(set, n, sum) : 
    
    # Base Cases 
    if (sum == 0) : 
        return True
    if (n == 0 and sum != 0) : 
        return False
   
    # If last element is greater than 
    # sum, then ignore it 
    if (set[n - 1] > sum) : 
        return isSubsetSum(set, n - 1, sum); 
   
    # else, check if sum can be obtained 
    # by any of the following 
    # (a) including the last element 
    # (b) excluding the last element    
    return isSubsetSum(set, n-1, sum) or isSubsetSum(set, n-1, sum-set[n-1]) 
      
      
# Driver program to test above function 
set = [3, 34, 4, 12, 5, 2] 
sum = 9
n = len(set) 
if (isSubsetSum(set, n, sum) == True) : 
    print("Found a subset with given sum") 
else : 
    print("No subset with given sum") 

#6############################################################################################

'''
m Coloring Problem

Given an undirected graph and a number m, determine if the graph can be coloured with at most 
m colours such that no two adjacent nodes of the graph are colored with the same color.

Input:  
graph = {0, 1, 1, 1},
        {1, 0, 1, 0},
        {1, 1, 0, 1},
        {1, 0, 1, 0}
Output: 
Solution Exists: 
Following are the assigned colors
 1  2  3  2
'''
class Graph():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]\
                              for row in range(vertices)]
 
    # A utility function to check 
    # if the current color assignment
    # is safe for vertex v
    def isSafe(self, v, colour, c):
        for i in range(self.V):
            if self.graph[v][i] == 1 and colour[i] == c:
                return False
        return True
     
    # A recursive utility function to solve m
    # coloring  problem
    def graphColourUtil(self, m, colour, v):
        if v == self.V:
            return True
 
        for c in range(1, m + 1):
            if self.isSafe(v, colour, c) == True:
                colour[v] = c
                if self.graphColourUtil(m, colour, v + 1) == True:
                    return True
                colour[v] = 0
 
    def graphColouring(self, m):
        colour = [0] * self.V
        if self.graphColourUtil(m, colour, 0) == None:
            return False
 
        # Print the solution
        print "Solution exist and Following 
                  are the assigned colours:"
        for c in colour:
            print c,
        return True
 
# Driver Code
g = Graph(4)
g.graph = [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
m = 3
g.graphColouring(m)

#7############################################################################################

'''
Hamiltonian Cycle

Hamiltonian Path in an undirected graph is a path that visits each vertex exactly once. A Hamiltonian 
cycle (or Hamiltonian circuit) is a Hamiltonian Path such that there is an edge (in the graph) from 
the last vertex to the first vertex of the Hamiltonian Path.

For example, a Hamiltonian Cycle in the following graph is {0, 1, 2, 4, 3, 0}.

(0)--(1)--(2)
 |   / \   |
 |  /   \  | 
 | /     \ |
(3)-------(4)

'''
class Graph():  
    def __init__(self, vertices):  
        self.graph = [[0 for column in range(vertices)] 
                            for row in range(vertices)]  
        self.V = vertices  # size of the graph
  
    def isSafe(self, v, pos, path):  
        # Check if current vertex and last vertex  
        # in path are adjacent  
        if self.graph[ path[pos-1] ][v] == 0:  
            return False
  
        # Check if current vertex not already in path  
        for vertex in path:  
            if vertex == v:  
                return False
  
        return True
  
    def hamCycleUtil(self, path, pos):  
  
        # base case: if all vertices are  
        # included in the path  
        if pos == self.V:  
            # Last vertex must be adjacent to the  
            # first vertex in path to make a cyle  
            if self.graph[ path[pos-1] ][ path[0] ] == 1:  
                return True
            else:  
                return False
  
        # Try different vertices as a next candidate  
        # in Hamiltonian Cycle. We don't try for 0 as  
        # we included 0 as starting point in hamCycle()  
        for v in range(1,self.V):  
  
            if self.isSafe(v, pos, path) == True:  
  
                path[pos] = v  
  
                if self.hamCycleUtil(path, pos+1) == True:  
                    return True
  
                # Remove current vertex if it doesn't  
                # lead to a solution  -- BACKTRACK
                path[pos] = -1
  
        return False
  
    def hamCycle(self):  
        path = [-1] * self.V  
  
        ''' Let us put vertex 0 as the first vertex  
            in the path. If there is a Hamiltonian Cycle,  
            then the path can be started from any point  
            of the cycle as the graph is undirected '''
        pos = 0
        path[pos] = 0

  
        if self.hamCycleUtil(path,pos+1) == False:  
            print ("Solution does not exist\n") 
            return False
  
        self.printSolution(path)  
        return True
  
    def printSolution(self, path):  
        print ("Solution Exists: Following", 
                 "is one Hamiltonian Cycle") 
        for vertex in path:  
            print (vertex, end = " ") 
        print (path[0], "\n") 
  
# Driver Code  
  
''' Let us create the following graph  
    (0)--(1)--(2)  
    | / \ |  
    | / \ |  
    | /  \ |  
    (3)-------(4) '''
g1 = Graph(5)  
g1.graph = [ [0, 1, 0, 1, 0], [1, 0, 1, 1, 1],  
            [0, 1, 0, 0, 1,],[1, 1, 0, 0, 1],  
            [0, 1, 1, 1, 0], ]  
  
# Print the solution  
g1.hamCycle();  
  
''' Let us create the following graph  
    (0)--(1)--(2)  
    | / \ |  
    | / \ |  
    | /  \ |  
    (3)  (4) '''
g2 = Graph(5)  
g2.graph = [ [0, 1, 0, 1, 0], [1, 0, 1, 1, 1],  
        [0, 1, 0, 0, 1,], [1, 1, 0, 0, 0],  
        [0, 1, 1, 0, 0], ]  
  
# Print the solution  
g2.hamCycle();  

#8############################################################################################

'''
Sudoku

Given a partially filled 9×9 2D array ‘grid[9][9]’, the goal is to assign digits (from 1 to 9) to 
the empty cells so that every row, column, and subgrid of size 3×3 contains exactly one instance 
of the digits from 1 to 9. 
'''
def print_grid(arr):
    for i in range(9):
        for j in range(9):
            print arr[i][j],
        print ('n')
 
         
# Function to Find the entry in 
# the Grid that is still  not used
# Searches the grid to find an 
# entry that is still unassigned. If
# found, the reference parameters 
# row, col will be set the location
# that is unassigned, and true is 
# returned. If no unassigned entries
# remains, false is returned.
# 'l' is a list  variable that has 
# been passed from the solve_sudoku function
# to keep track of incrementation 
# of Rows and Columns
def find_empty_location(arr, l):
    for row in range(9):
        for col in range(9):
            if(arr[row][col]== 0):
                l[0]= row
                l[1]= col
                return True
    return False
 
# Returns a boolean which indicates 
# whether any assigned entry
# in the specified row matches 
# the given number.
def used_in_row(arr, row, num):
    for i in range(9):
        if(arr[row][i] == num):
            return True
    return False
 
# Returns a boolean which indicates 
# whether any assigned entry
# in the specified column matches 
# the given number.
def used_in_col(arr, col, num):
    for i in range(9):
        if(arr[i][col] == num):
            return True
    return False
 
# Returns a boolean which indicates 
# whether any assigned entry
# within the specified 3x3 box 
# matches the given number
def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if(arr[i + row][j + col] == num):
                return True
    return False
 
# Checks whether it will be legal 
# to assign num to the given row, col
# Returns a boolean which indicates 
# whether it will be legal to assign
# num to the given row, col location.
def check_location_is_safe(arr, row, col, num):
     
    # Check if 'num' is not already 
    # placed in current row,
    # current column and current 3x3 box
    return not used_in_row(arr, row, num) and
           not used_in_col(arr, col, num) and
           not used_in_box(arr, row - row % 3, # you need to check each box starting from top most left most corner
                           col - col % 3, num) # same for column
 
# Takes a partially filled-in grid 
# and attempts to assign values to
# all unassigned locations in such a 
# way to meet the requirements
# for Sudoku solution (non-duplication 
# across rows, columns, and boxes)
def solve_sudoku(arr):
     
    # 'l' is a list variable that keeps the 
    # record of row and col in 
    # find_empty_location Function    
    l =[0, 0]
     
    # If there is no unassigned 
    # location, we are done    
    if(not find_empty_location(arr, l)):
        return True
     
    # Assigning list values to row and col 
    # that we got from the above Function 
    row = l[0]
    col = l[1]
     
    # consider digits 1 to 9
    for num in range(1, 10):
         
        # if looks promising
        if(check_location_is_safe(arr, 
                          row, col, num)):
             
            # make tentative assignment
            arr[row][col]= num
 
            # return, if success, 
            # ya ! 
            if(solve_sudoku(arr)):
                return True
 
            # failure, unmake & try again
            arr[row][col] = 0
             
    # this triggers backtracking        
    return False
 
# Driver main function to test above functions
if __name__=="__main__":
     
    # creating a 2D array for the grid
    grid =[[0 for x in range(9)]for y in range(9)]
     
    # assigning values to the grid
    grid =[[3, 0, 6, 5, 0, 8, 4, 0, 0],
          [5, 2, 0, 0, 0, 0, 0, 0, 0],
          [0, 8, 7, 0, 0, 0, 0, 3, 1],
          [0, 0, 3, 0, 1, 0, 0, 8, 0],
          [9, 0, 0, 8, 6, 3, 0, 0, 5],
          [0, 5, 0, 0, 9, 0, 6, 0, 0],
          [1, 3, 0, 0, 0, 0, 2, 5, 0],
          [0, 0, 0, 0, 0, 0, 0, 7, 4],
          [0, 0, 5, 2, 0, 6, 3, 0, 0]]
     
    # if success print the grid
    if(solve_sudoku(grid)):
        print_grid(grid)
    else:
        print "No solution exists"

