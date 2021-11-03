'''
sort using python lib
'''
def sort(touples):
  return sorted(touples, key=lambda touples: (touples[0], touples[1]))

#Test Case
touples = ((2,3), (4,1), (1,9), (9,0), (1,4), (2,2), (2,1), (5,7))
print(sort(touples))

#1##############################################################################

'''
Merge sort   time : O(n lgn)    space: O(n)
'''
def mergeSort(arr):
    n = len(arr)
    if n == 1:
        return arr
    m = int(n/2)
    l = mergeSort(arr[:m])
    r = mergeSort(arr[m:])    
    return merger(l, r)

def merger(l, r):
    len_l = len(l)
    len_r = len(r)
    j = 0
    i = 0
    temp = []
    while i < len_l and j < len_r:
        if l[i] <= r[j]:
            temp.append(l[i])
            i += 1
        else:
            temp.append(r[j])
            j += 1
    while i < len_l:
        temp.append(l[i])
        i += 1
    while j < len_r:
        temp.append(r[j])
        j += 1
        
    return temp

#test case
arr = [38, 27, 43, 3, 9, 82, 10]
print("Merge Sort Result : ", mergeSort(arr))

#2##############################################################################

'''
heap sorted     time : O(n lgn)    space: O(1)
'''
def heapSort(arr):
    n = len(arr)
    #max heap
    for i in range(n,-1,-1):
        heapify(arr, i, n)
    
    #keep largest at the end
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, 0, i)
    
    return arr


    
def heapify(arr, i, n):    
    largest_index = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[largest_index] < arr[l]:
        largest_index = l
    if r < n and arr[largest_index] < arr[r]:
        largest_index = r
    if largest_index != i:
        arr[largest_index], arr[i] = arr[i], arr[largest_index]
        heapify(arr, largest_index, n)
    
    
#test cases       
arr = [9,2,5,3,8,10,6,5,4,7]
print(heapSort(arr))    

#3##############################################################################

'''
find unique premutations of a set
'''
from itertools import permutations
x = [1,1,0,0]
perm = set(permutations(x))

print(perm)

#4##############################################################################

'''
find all subsets
'''

from itertools import chain, combinations
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        return list(chain.from_iterable(combinations(nums, i) for i in range(len(nums)+1)))
#5##############################################################################

'''
Tree problems
'''
class Node :
    def __init__(self,data) :
        self.data = data
        self.right = None
        self.left = None

class Tree :
    def __init__(self, data) :
        self.root = Node(data)
        
    def height(self, currentNode) :
        if currentNode == None :
            return 0
        return 1 + max(self.height(currentNode.right), self.height(currentNode.left))
    
    def isBalanced(self,currentNode):
        # a None tree is balanced
        if currentNode is None:
            return True
        return (self.isBalanced(currentNode.right) and self.isBalanced(currentNode.left) and
                abs(self.height(currentNode.left) - self.height(currentNode.right)) <= 1)

#test cases
tree = Tree(1)
tree.root.left = Node(2)
tree.root.left.right = Node(4)
tree.root.left.right.left = Node(6)
tree.root.right = Node(3)
tree.root.right.right = Node(5)
tree.root.right.right.right = Node(7)


print("Tree height is : ", tree.height(tree.root))

print("Tree is balanced? ", tree.isBalanced(tree.root))        

#6##############################################################################

'''
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root.

Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.

Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

 

Example 1:


Input: root = [3,2,3,null,3,null,1]
Output: 7
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
'''

class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(root):
            if not root:
                return [0,0]
            l = dfs(root.left)
            r = dfs(root.right)
            return [root.val + l[1] + r[1], max(l) + max(r)]
        return max(dfs(root))

#7##############################################################################
'''
Graph problems
'''

class Graph:

    def __init__(self, graph) :
        self.graph = graph
        self.numOfVertices = len(graph)
        self.isVisited = [False] * self.numOfVertices
        self.adjVertices = [[] for i in range(self.numOfVertices)]
        self.adjWeights = [[] for i in range(self.numOfVertices)]

        for i in range(self.numOfVertices) :
            index = 0
            for w in self.graph[i] :
                if w> 0 :
                    self.adjVertices[i].append(index)
                    self.adjWeights[i].append(w)
                index += 1
        self.mazeSolution = [[0 for i in range(self.numOfVertices)] for j in range(self.numOfVertices)]
        self.nQueen = [[0 for i in range(self.numOfVertices)] for j in range(self.numOfVertices)]

    def isCyclical(self, vertice):
        visited = [False] * self.numOfVertices
        stack = [False] * self.numOfVertices
        for v in list(self.graph.keys()) :
            #print(v)
            if visited[v] == False :
                if self.checkCycle(v, visited, stack) == True :
                    return True
        return False

    def checkCycle(self,vertice, visited, stack) :
        visited[vertice] = True
        stack[vertice] = True
        for v in self.graph[vertice] :
            if visited[v] == False :
                if self.checkCycle(v, visited, stack) == True :
                    return True
                elif stack[v] == True :
                    return True
        return False

    def dfs(self, vertice) :
        self.isVisited[vertice] = True
        print(vertice)
        for v in self.graph[vertice] :
            if self.isVisited[v] == False :
                self.dfs(v)
            break
        self.isVisited = [False] * self.numOfVertices#reset isVisited

    def bfs(self, vertice) :
        self.isVisited[vertice] = True
        queue = []
        queue.append(vertice)
        while queue :
            s = queue.pop(0)
            print(s)
            for v in self.graph[s] :
                if self.isVisited[v] == False :
                    queue.append(v)
                    self.isVisited[v] = True
        self.isVisited = [False] * self.numOfVertices
    def dijkstra(self, vertice) :
        maxValue = 1000#a very large number, consider infinity compared to all weights
        shortestPath = [maxValue] * self.numOfVertices
        shortestPath[vertice] = 0

        for loop in range(self.numOfVertices) :

            minValue = maxValue
            minValueIndex = -1
            for v in range(self.numOfVertices) :
                if self.isVisited[v] == False and shortestPath[v] < minValue :
                        minValue = shortestPath[v]
                        minValueIndex = v

            self.isVisited[minValueIndex] = True

            for v in range(self.numOfVertices) :
                if self.graph[minValueIndex][v] > 0 and self.isVisited[v] == False and \
                shortestPath[v] > shortestPath[minValueIndex] + self.graph[minValueIndex][v] :
                    shortestPath[v] = shortestPath[minValueIndex] + self.graph[minValueIndex][v]

        print(shortestPath)

    def isPathForDistance(self, path, distance) :
        currentVertice = path[-1]
        for index in range(len(self.adjVertices[currentVertice])) :
            v = self.adjVertices[currentVertice][index]
            w = self.adjWeights[currentVertice][index]
            if v in path :
                continue #ignore vertice if it was seen befor
            if w >= distance :
                print(path)
                return True
            path.append(v)
            if self.isPathForDistance(path, distance-w) :
                return True
            #backtrack
            path.pop()
        return False

    def solveMaze(x, y, n, maze, sol):
        if x == n-1 and y == n-1:
            sol[x][y] = 1
            return True

        if x >= 0 and x < n and y >=0 and y < n and maze[x][y] == 1:
            sol[x][y] = 1

            if solveMaze(x+1, y, n, maze, sol):
                return True

            if solveMaze(x, y+1, n, maze, sol):
                return True

            #backtrack
            sol[x][y] = 0
        print(sol)
        return False

    def nQueenProblem(self, col, numOfQueens) :
        if numOfQueens == 0 :
            return True

        for row in range(self.numOfVertices) :
            if self.isNQueenSafeMove(row, col) :
                self.nQueen[row][col] = 1
                numOfQueens -= 1
                if self.nQueenProblem(col+1, numOfQueens) :
                    return True

                #backtrack
                self.nQueen[row][col] = 0
                numOfQueens += 1


        return False

    def isNQueenSafeMove(self,row, col) :
        #check row/col
        for i in range(col) :
            if self.nQueen[row][i] == 1 :
                return False
        #check diagonal - upper left
        i=row
        j=col
        while i >= 0 and j >= 0 :
            if self.nQueen[i][j] == 1 :
                return False
            i -= 1
            j -= 1
        #check diagonal - lupper right
        i=row
        j=col
        while i < self.numOfVertices and j != 0 :
            if self.nQueen[i][j] == 1 :
                return False
            i += 1
            j -= 1

        return True

    def hamiltonianPathExists(self, startingVertice, path,graph) :
        if startingVertice not in range(self.numOfVertices) :
            return False
        if len(path) == self.numOfVertices and self.graph[path[0]][path[-1]] == 1 :
            print("Path exisits and it is : ",path)
            return True
        if self.isVisited[startingVertice] == False :
            self.isVisited[startingVertice] = True
            path.append(startingVertice)
        #for vertice in range(self.numOfVertices) :

        for vertice in self.adjVertices[startingVertice] :
            if self.isVisited[vertice] == False :
                path.append(vertice)
                self.isVisited[vertice] = True

                if self.hamiltonianPathExists(path[-1], path, graph) :
                    return True

                #backtrack
                self.isVisited[path.pop()] = False
        return False

#8##############################################################################

'''
Travelling salesman problem is a Hamiltonian cycle with min cost. the bruteforce solution
is costly O(N!). 
to make it dynamic, we want to have distinct states. for a 4 node problem, 
mask should vary between 0000 and 1111; that's 2^N possibilities. O(2^N . N)

'''
class TravellingSalesman:
    def __init__(self, graph, s):
        self.graph = graph
        self.N = len(graph)
        self.VISISTED_ALL = (1<<self.N) -1  # 2^N
        self.S = s#starting city
        self.MAX = 9999
        self.loockup = []
        for i in range(1<<self.N):
            new = []
            for j in range(self.N):
                new.append(-1)
            self.loockup.append(new) #loockup[2^N][N] = -1
        #print(self.dynamic)
        
    def minCost(self, mask, pos):
        #if all nodes were visited (mask 1101 shows that node B is not visited yet)
        if mask == self.VISISTED_ALL:
            return self.graph[pos][self.S]
        
        #lookup checkpoint, to make the program dynamic
        if self.loockup[mask][pos] != -1: #if the value has changed already
            return self.loockup[mask][pos]
        
        ans = self.MAX
        #traverse unvisited cities
        for city in range(self.N):
            
            #if current city is not visited
            if (mask & (1 << city)) == 0: # &: if L>R = 0; L<=R = L;
                #distance from current city to new city, plus remaining distance 
                #(mark city as visited in mask in recursion)
                
                # suppose cities are A, B, C and D. Let's have starting point at A. Mask would be 0001.
                # going to city B, we'll do 1 left shift in city, i.e. 0010. the OR result with mask
                # is (0001 | 0010) = 0011, meaning that A and B are visited.
                
                newAns = self.graph[pos][city] + self.minCost(mask|(1<<city), city) 
                ans = min(ans, newAns)
        self.loockup[mask][pos] = ans
        return ans
    
    
distances=[
        [0, 20, 42, 25],
        [20, 0, 30, 34],
        [42, 30, 0, 10],
        [25, 34, 10, 0]
        ]    
ts = TravellingSalesman(distances, 0)
print("Travelling Salesman Cost = ", ts.minCost(1,0)) #mask=1 means city is visited, pos=0 means we start from city 0    


#9##############################################################################

'''
Count total ways to reach destination from source in an undirected Graph. 

using depth first traverse we track the visited nodes to find the path. everytime
we calculate pathCount in recursion we visit and unvisit the node because we should
consider the node once we go through it for one path, but then we should unvisit it
to consider other paths that goes thrugh this node.

O(n^2)
'''
def countofPossiblePaths(mtrx, mtrx_len, src, dest):
    visited = [False] * (mtrx_len-1)
    visited[src] = True
    return countofPossiblePaths_util(mtrx, mtrx_len-1, src, dest, visited)
    
def countofPossiblePaths_util(mtrx, mtrx_len, src, dest, visited):
    if src == dest:
        return 1
    pathCount = 0
    for v in range(mtrx_len):
        if mtrx[src][v] == 1 and visited[v] == False:
            visited[v] = True
            pathCount += countofPossiblePaths_util(mtrx, mtrx_len, v, dest, visited)
            visited[v] = False #make source not visited for backtracking
    return pathCount


mtrx = [ 
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0 ], 
        [ 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0 ], 
        [ 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 ], 
        [ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ], 
        [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 ], 
        [ 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0 ], 
        [ 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0 ], 
        [ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0 ], 
        [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0 ], 
        [ 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0 ], 
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ] 
    ] 

print("Number of possible pathes between 2 and 8 = ", countofPossiblePaths(mtrx, len(mtrx), 2, 8))

#10##############################################################################

'''
Given an expression string exp , write a program to examine whether the pairs
and the orders of “{“,”}”,”(“,”)”,”[“,”]” are correct in exp.
Input: exp = “[()]{}{[()()]()}”
Output: Balanced

Input: exp = “[(])”
Output: Not Balanced
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:42:03 2020

@author: mehrdadalemzadeh
"""

def isBalanced(exp):
    stack = []
    for e in exp:
        if e in ["[", "{", "("]:
            stack.append(e)
        elif e in ["]", "}", ")"]:
            temp = stack.pop()
            if (temp == "(" and e != ")") or (temp == "{" and e != "}") or (temp == "[" and e != "]"):
                return False
    if len(stack) == 0:
        return True
    else:
        return

#Test Case
exp1 = "[()]{}{[()()]()}"
exp2 = "[(])"
print("Expression 1 is balanced? ", isBalanced(exp1))
print("Expression 2 is balanced? ", isBalanced(exp2))

#11##############################################################################

def HanoiTower(n, Tower1, Tower2, Tower3):
  if n==1:
    print ("Move disk",n,"from tower",Tower1,"to tower",Tower2)
    return
  HanoiTower(n-1, Tower1, Tower3, Tower2)
  print ("Move disk",n,"from tower",Tower1,"to tower",Tower2)
  HanoiTower(n-1, Tower3, Tower2, Tower1)


#Test Case
n=3
HanoiTower(n, 'A', 'B', 'C')

#12##############################################################################

"""
Minimum possible travel cost among N cities

There are N cities situated on a straight road and each is separated by a distance of 1 unit.
You have to reach the (N + 1)th city by boarding a bus. The ith city would cost of C[i] dollars
to travel 1 unit of distance. In other words, cost to travel from the ith city to the jth city
is abs(i – j ) * C[i] dollars. The task is to find the minimum cost to travel from city 1 to
city (N + 1) i.e. beyond the last city.

Input: CityCost = {4, 7, 8, 3, 4}
Output: 18
Board the bus at the first city then change
the bus at the fourth city.
(3 * 4) + (2 * 3) = 12 + 6 = 18
"""

def minTravelCost(citiesCost):
    n = len(citiesCost)
    boardingStation = 0
    cost = 0
    for i in range(1,n):
        if citiesCost[i] < citiesCost[boardingStation]:
            cost += citiesCost[boardingStation] * (i-boardingStation)
            boardingStation = i
    cost += citiesCost[boardingStation] * (n - boardingStation)
    return cost

#Test Cases
citiesCostArray1 = [3, 5, 4]
citiesCostArray2 = [4, 7, 8, 3, 4]
print("minTravelCost 1 = ", minTravelCost(citiesCostArray1))
print("minTravelCost 2 = ", minTravelCost(citiesCostArray2))

#13##############################################################################

import sys
class Matrix :
    def __init__(self, matrix) :
        self.matrix = matrix
        self.rowSize = len(matrix)
        self.colSize = len(matrix[0])
    def minCostPath(self, rowDest, colDest):
        if rowDest < 0 or colDest < 0 :
            return sys.maxsize
        if rowDest == 0 and colDest == 0 :
            return self.matrix[0][0]
        else:
            return self.matrix[rowDest][colDest] + min(self.minCostPath(rowDest-1,colDest-1),
                              self.minCostPath(rowDest-1,colDest),self.minCostPath(rowDest,colDest-1))
    def minNumberOfMoves(self, beginX, beginY, endX, endY) :
        if beginX == endX and beginY == endY :
            return 0
        if endX < 0 or endY < 0 :
            return 0
        else:
            return 1 + min(self.minNumberOfMoves(beginX, beginY, endX-1, endY-1) +
                           self.minNumberOfMoves(beginX, beginY, endX-1, endY) +
                           self.minNumberOfMoves(beginX, beginY, endX, endY-1))




matrix = [
            [1,3,5,8],
            [4,2,1,7],
            [4,3,2,3]
         ]

m = Matrix(matrix)
m.minCostPath(2,3)

m.minNumberOfMoves(0,0,2,3)


#14##############################################################################

'''
Find day of the week for a given date

https://www.youtube.com/watch?v=dD8iXe_InnQ
'''
def dayofweek(d, m, y): 
    t = [ 0, 3, 2, 5, 0, 3, 
          5, 1, 4, 6, 2, 4 ] 
    y -= m < 3
    return (( y + int(y / 4) - int(y / 100) 
             + int(y / 400) + t[m - 1] + d) % 7) 
  
# Driver Code 
day = dayofweek(30, 8, 2010) 
print(day) 

#15##############################################################################

'''
Add two numbers without using arithmetic operators

Sum of two bits can be obtained by performing XOR (^) of the two bits. Carry bit can be obtained by performing AND (&) of two bits. 
Above is simple Half Adder logic that can be used to add 2 single bits. We can extend this logic for integers. If x and y don’t 
have set bits at same position(s), then bitwise XOR (^) of x and y gives the sum of x and y. To incorporate common set bits also, 
bitwise AND (&) is used. Bitwise AND of x and y gives all carry bits. We calculate (x & y) << 1 and add it to x ^ y to get the required result. 
'''
# without using arithmetic operator
def Add(x, y):
 
    # Iterate till there is no carry 
    while (y != 0):
     
        # carry now contains common
        # set bits of x and y
        carry = x & y
 
        # Sum of bits of x and y where at
        # least one of the bits is not set
        x = x ^ y
 
        # Carry is shifted by one so that   
        # adding it to x gives the required sum
        y = carry << 1
     
    return x
 
print(Add(15, 32))

#16##############################################################################

'''
How to check if a given number is Fibonacci number?

A simple way is to generate Fibonacci numbers until the generated number is greater than or equal to ‘n’. 
Following is an interesting property about Fibonacci numbers that can also be used to check if a given 
number is Fibonacci or not. A number is Fibonacci if and only if one or both of (5*n^2 + 4) or (5*n^2 – 4) 
is a perfect square
'''

import math 
  
# A utility function that returns true if x is perfect square 
def isPerfectSquare(x): 
    s = int(math.sqrt(x)) 
    return s*s == x 
  
# Returns true if n is a Fibinacci Number, else false 
def isFibonacci(n): 
  
    # n is Fibinacci if one of 5*n*n + 4 or 5*n*n - 4 or both 
    # is a perferct square 
    return isPerfectSquare(5*n*n + 4) or isPerfectSquare(5*n*n - 4) 
     
# A utility function to test above functions 
for i in range(1,11): 
     if (isFibonacci(i) == True): 
         print i,"is a Fibonacci Number"
     else: 
         print i,"is a not Fibonacci Number "


#17##############################################################################
'''
DP
'''

def fibo(n):
    memo = [None] * n
    return fibo_util(n, memo)

def fibo_util(n, memo):
    if memo[n-1] != None:
        return memo[n-1]
    if n == 1 or n == 2:
        return 1

    result = fibo(n-1) + fibo(n-2)
    memo[n-1] = result

    return result

print(fibo(10))    


#18##############################################################################

def knapsack(v, w, k, n): # O(2^n)
    if k <= 0 or n <= 0:
        return 0
    if w[n-1] > k:
        return knapsack(v, w, k, n-1)
    else:
        return max(
            knapsack(v, w, k, n-1),
            knapsack(v, w, k - w[n-1], n-1) + v[n-1]
        )



def knapsack_dp2(v, w, c, n):
    if memo[n][c] != 0:
        return memo[n][c]
    if n <= 0 or c <= 0:
        return 0
    if w[n-1] > c:
        return knapsack_dp2(v, w, c, n-1)
    else:
        memo[n][c] = max(
            knapsack_dp2(v, w, c, n-1),
            knapsack_dp2(v, w, c - w[n-1], n-1) + v[n-1]
        )
    return memo[n][c]


v = [60, 100, 120] 
w = [10, 20, 30] 
c = 49
n = len(v) 
memo = [[0 for i in range(c+1)] for j in range(n+1)]
print("Knapsack Recursize answer : ", knapsack(v, w, c, n))
print("Knapsack DP answer : ", knapsack_dp2(v, w, c, n))

#19##############################################################################

'''
Hashing 

given two strings compare to see if there are common substrings between them

sol: create a hash array from all alphabets. go through the first string and mark the alphabet as true then
go through the second and compare
'''

def commonSubstrings(s1, s2) : 
    MAX_CHAR = 26
    v = [0] * (MAX_CHAR) 
      
    #ord gives the ascci number of a char
    for i in range(len(s1)): 
        v[ord(s1[i]) - ord('a')] = True
      
    # checking common substring  
    # of str2 in str1 
    for i in range(len(s2)) : 
        if (v[ord(s2[i]) - ord('a')]) : 
            return True
      
    return False

#20##############################################################################

'''
Job sequencing O(n^2) - greedy algo

greedy algorithm

we get a list of jobs with IDs, deadline (in unit of time), and profit.

JobID  Deadline  Profit
  a      4        20   
  b      1        10
  c      1        40  
  d      1        30
  
we want to maximize profit if only one job can be scheduled at a time.

solution is to sort the table based on profit. 
take the highest value and check the deadline. if it's 3 for instance, you have to put the ID
in an array on the third slot. if third slot is full, go one back until you cannot go further back.
if there is no place to put it then we should skip the job  
'''

def jobSequencing(jobs):
    n = len(jobs)
    temp_sequence = [0 for i in range(n)]
    jobs_sorted = sortJobsOnProfit(jobs)

    for i in range(n-1, -1, -1):

        j = jobs_sorted[i][1] - 1
      
        while j >= 0 :
            if temp_sequence[j] == 0:
                temp_sequence[j] = jobs_sorted[i][0]
                break
            j -= 1
    sequence = []
    for e in temp_sequence :
        if e != 0:
            sequence.append(e)
    return sequence
    
def sortJobsOnProfit(jobs):
    return sorted(jobs, key = lambda j : j[2])
    
    
jobs = [['a',4,20], 
        ['b', 1, 10],
        ['c', 1, 40],
        ['d', 1, 30]
        ]    

print("Job Sequence to Maximize Profit : ", jobSequencing(jobs))

###############################################################################