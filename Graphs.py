'''

https://www.geeksforgeeks.org/a-search-algorithm/

https://www.geeksforgeeks.org/implementing-water-supply-problem-using-breadth-first-search/?ref=leftbar-rightbar

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

    def solveMaze(self, x, y) :

        if x == self.numOfVertices - 1 and y == self.numOfVertices - 1 :
            self.mazeSolution[x][y] = 1
            return True

        if x >= 0 and x < self.numOfVertices and y >= 0 and y < self.numOfVertices and maze[x][y] == 1:
            self.mazeSolution[x][y] = 1

            if self.solveMaze(x+1, y) :
                return True

            if self.solveMaze(x, y+1) :
                return True

            #backtrack
            self.mazeSolution[x][y] = 0
        print(self.mazeSolution)
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
        #check diagonal - lower left
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

    def countPathsBetween(self,v1, v2) :
        stack = []
        path = []
        count = 0

        path.append(v1)
        stack.append(v1)
        self.isVisited[v1] = True

        while stack :
            for v in self.graph[stack.pop(0)]:
                if self.isVisited[v] == False :
                    path.append(v)
                    stack.append(v)
                    if v != v2:
                        self.isVisited[v] = True
        for v in path :
            if v == v2 :
                count += 1
        print(path)
        return count



print("----------Test Cases : ---------")

graph = {0:[1,2], 1:[4,3], 2:[0], 3:[1], 4:[3]}#input for checking cycle, bfs, dfs

weightedGraph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
        [4, 0, 8, 0, 0, 0, 0, 11, 0],
        [0, 8, 0, 7, 0, 4, 0, 0, 2],
        [0, 0, 7, 0, 9, 14, 0, 0, 0],
        [0, 0, 0, 9, 0, 10, 0, 0, 0],
        [0, 0, 4, 14, 10, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 1, 6],
        [8, 11, 0, 0, 0, 0, 1, 0, 7],
        [0, 0, 2, 0, 0, 0, 6, 7, 0]
        ]

g = Graph(graph)
'''
cycle detection algorithm :
    1. start from the given node
    2. which ever node you turn into:
    3. if it wans't visited then mark it as visited and push it onto the stack
    4. if it was visited, check stack to see if it exisits
        4.1. if it exisits then you have a cycle
        4.2. else, leave it and continue with the next node
'''
print("Graph has cycle:", g.isCyclical(0))
print("'-------DFS---------")
g.dfs(2)
print("'-------BFS---------")
g.bfs(0)
print("'-------Dijkstra---------")
wg = Graph(weightedGraph)
wg.dijkstra(0)
wg.isVisited = [False] * 8
print("-----is there a path with min distance abc?------")
'''
The idea is to use Backtracking. We start from given source, explore all paths from current
vertex. We keep track of current distance from source. If distance becomes more than k, we
return true. If a path doesn’t produces more than k distance, we backtrack.

How do we make sure that the path is simple and we don’t loop in a cycle? The idea is to keep
track of current path vertices in an array. Whenever we add a vertex to path, we check if it
already exists or not in current path. If it exists, we ignore the edge.
'''
print(wg.isPathForDistance([0], 57))



print("---------Maze Solution----------")

maze = [ [1, 0, 0, 0],
         [1, 1, 0, 1],
         [0, 1, 0, 0],
         [1, 1, 1, 1] ]

m = Graph(maze)
print(m.solveMaze(0,0))



print("---------N Queen Solution----------")

chess = [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]
         ]

q = Graph(chess)
q.nQueenProblem(0,7)
for boardRows in q.nQueen :
    print(boardRows)

print("---------hamiltonian Path----------")
'''
Let us create the following graph
      (0)--(1)--(2)
       |   / \   |
       |  /   \  |
       | /     \ |
      (3)-------(4)
'''
graph1 = [ [0, 1, 0, 1, 0], [1, 0, 1, 1, 1],  [0, 1, 0, 0, 1,], [1, 1, 0, 0, 0],  [0, 1, 1, 0, 0], ]

'''
Let us create the following graph
      (0)--(1)--(2)
       |   / \   |
       |  /   \  |
       | /     \ |
      (3)       (4)
'''
graph2 = [ [0, 1, 0, 1, 0], [1, 0, 1, 1, 1],  [0, 1, 0, 0, 1,],[1, 1, 0, 0, 1],  [0, 1, 1, 1, 0], ]

h1 = Graph(graph1)
print("does hamiltonian path exist? ",h1.hamiltonianPathExists(0,[], 1))
print("-------------------------------------")
h2 = Graph(graph2)
print("does hamiltonian path exist? ",h2.hamiltonianPathExists(0,[],2))


graph3 = {0:[1,3], 1:[2,3,4], 2:[1,4], 3:[0,1,4], 4:[2,1,3]}
p = Graph(graph3)
print("count num of paths = ", p.countPathsBetween(0,4))


################################################################################

'''
find out if a graph is a tree too
sol:
    check 1 - if there is a cycle: if v has a connection to u and u is a parent of v, directly or over connection
    check 2 - all nodes should be connected: if all nodes are visited
    
'''

def isTree(graph, verticeCount):           
    #if you start from any node you may not find the entire graph. you need to start from every node and
    #if only one result comes true then the graph is a tree
    finalDecision = []
    for i in range(verticeCount):
        visited = [False] * verticeCount 
        finalDecision.append(isTree_util(graph, visited, i, [], []))
    print(finalDecision)        
    return True in finalDecision
    
def isTree_util(graph, visited, v, stack, cycleCheck):
    visited[v] = True
    cycleCheck.append(v)
    #print(v)
    if v in graph.keys():#make sure graph has a key like v before checking graph[v]
        for u in graph[v]:#add all children to stack
            stack.append(u)
            if u not in cycleCheck:
                cycleCheck.append(u)
    #print(stack)
    while len(stack) != 0:
        parent = stack.pop()
        if visited[parent] == False:
            isTree_util(graph, visited, parent, stack, cycleCheck)
            
    allVerticesVisited = False
    if False not in visited:
        allVerticesVisited = True  

    #print(cycleCheck)
    #print(visited)          

    return hasDuplicate(cycleCheck) and allVerticesVisited
    
def hasDuplicate(cycleCheck):
    map = {};
    for e in cycleCheck:
        if e in map.keys():
            return True
        else:
            map[e] = 1
    return False
    
    
                
    
#test case
graph1 = {0:[2,3], 4:[0], 3:[4]}    
visited = [False] * 5    
print("Graph1 is a tree? ", isTree(graph1, 5))

graph2 = {0:[2,3], 1:[0], 3:[4]}    
visited = [False] * 5    
print("Graph2 is a tree? ", isTree(graph2, 5))

################################################################################

def dfs(graph, verticeCount):
    visited = [False] * verticeCount
    #you can start from any v. here we are starting from 0
    dfs_util(graph, visited, 0, [])

def dfs_util(graph, visited, v, stack):
    visited[v] = True
    print(v)
    if v in graph.keys():#make sure graph has a key like v before checking graph[v]
        for u in graph[v]:#add all children to stack
            stack.append(u)
    #print(stack)
    while len(stack) != 0:
        parent = stack.pop()
        if visited[parent] == False:
            dfs_util(graph, visited, parent, stack)    
            
graph1 = {0:[2,3], 4:[0], 3:[4]}    
print("Graph1 DFS : ")
dfs(graph1, 5)

graph2 = {0:[2,3], 1:[0], 3:[4]}    
print("Graph2 DFS : ")
dfs(graph2, 5)

################################################################################

'''
DFS: find a node given the starting position
'''    
tree2 = {'S': ['A', 'B', 'C'],
        'A': ['S', 'D', 'E', 'G'],
        'B': ['S', 'G'],
        'C': ['S', 'G'],
        'D': ['A'],
        'E': ['A']}

tree = {'S': ['A', 'B'],
         'A': ['S'],
         'B': ['S', 'C', 'D'],
         'C': ['B', 'E', 'F'],
         'D': ['B', 'G'],
         'E': ['C'],
         'F': ['C']
         }    

def DFS(array, find):
    global visited_list, tree
    # if all the children of current node has already been visited
    # or has no children, dequeue the last node in array and rerun
    # function DFS
    if set(tree[array[-1]]).issubset(visited_list):
        del array[-1]
        return DFS(array, find)

    for item in tree[array[-1]]:
        if item in visited_list:
            continue
        visited_list.append(item)
        array.append(item)
        if item == find:
            return array, visited_list
        else:
            return DFS(array, find)


if __name__ == '__main__':
    visited_list = ['S']
    find = 'G'
    solution, visited_nodes = DFS(['S'], find)
    print('Optimal solution: ' + str(solution))
    print('visited nodes: ' + str(visited_nodes))

################################################################################


'''
BFS: find a node given the starting position
'''    
tree = {'S': ['A', 'B', 'C'],
        'A': ['S', 'D', 'E', 'G'],
        'B': ['S', 'G'],
        'C': ['S', 'G'],
        'D': ['A'],
        'E': ['A']}

tree2 = {'S': ['A', 'B'],
         'A': ['S'],
         'B': ['S', 'C', 'D'],
         'C': ['B', 'E', 'F'],
         'D': ['B', 'G'],
         'E': ['C'],
         'F': ['C']
         }


def BFS(array):
    global tree2
    index = 0               # record steps needed to achieve goal node
    nodes_layers = [['S']]  # record nodes in all visited layers
    solution = ['G']
    current_target = 'G'    # current node to find reversely in order to find the optimal path

    # Get visited nodes sequence
    while 'G' not in array:
        temp = []   # record nodes in each layer
        for item in tree2[array[index]]:
            if item in array:
                continue
            temp.append(item)
            array.append(item)
            if item == 'G':     # stop the loop if goal is finded
                break
        nodes_layers.append(temp)
        index += 1

    # get optimal path, starting from goal
    for i in range(index-1, 0, -1):
        for j in range(len(nodes_layers[i])):
            if current_target in tree2[nodes_layers[i][j]]:
                current_target = nodes_layers[i][j]
                solution.append(nodes_layers[i][j])
                break
    solution.append('S')    # append starting node to solution
    solution.reverse()      # reverse the solution array to get the path
    return solution, array


if __name__ == '__main__':
    solution, nodes_visited = BFS(['S'])
    print('Optimal solution: ' + str(solution))
    print('Visited nodes: ' + str(nodes_visited))

################################################################################


'''
A* Search algorithm is one of the best and popular technique used in 
path-finding and graph traversals. A* algorithm is an advanced form 
of BFS. 

We can consider a 2D Grid having several obstacles and we start from 
a source cell (coloured red below) to reach towards a goal cell 
(coloured green below)

What A* Search Algorithm does is that at each step it picks the node 
according to a value-‘f’ which is a parameter equal to the sum of two 
other parameters – ‘g’ and ‘h’. At each step it picks the node/cell 
having the lowest ‘f’, and process that node/cell.

We define ‘g’ and ‘h’ as simply as possible below

g = the movement cost to move from the starting point to a given square 
on the grid, following the path generated to get there.
h = the estimated movement cost to move from that given square on the 
grid to the final destination. This is often referred to as the heuristic, 
which is nothing but a kind of smart guess. We really don’t know the 
actual distance until we find the path, because all sorts of things 
can be in the way (walls, water, etc.). Heurestic can be done using many 
ways: i.e. Manhattan, Diagonal and Euclidean. 

Dijkstra is a special case of A*, where h = 0 for all nodes.

A) Exact Heuristics –

we can also find the exact h but it's time consuming. 2 methods are

1) Pre-compute the distance between each pair of cells before running 
the A* Search Algorithm.
2) If there are no blocked cells/obstacles then we can just find the exact 
value of h without any pre-computation using the distance formula/Euclidean 
Distance    

B) Approximation Heuristics –

1) Manhattan Distance –
h = abs (current_cell.x – goal.x) + abs (current_cell.y – goal.y) 
When to use this heuristic? – When we are allowed to move only in four 
directions only (right, left, top, bottom)

2) Diagonal Distance-

h = max { abs(current_cell.x – goal.x), abs(current_cell.y – goal.y) } 
When to use this heuristic? – When we are allowed to move in eight directions 
only (similar to a move of a King in Chess)

3) Euclidean Distance-

h = sqrt ( (current_cell.x – goal.x)2 + (current_cell.y – goal.y)2 ) 
When to use this heuristic? – When we are allowed to move in any directions.

'''

tree = {'S': [['A', 1], ['B', 5], ['C', 8]],
        'A': [['S', 1], ['D', 3], ['E', 7], ['G', 9]],
        'B': [['S', 5], ['G', 4]],
        'C': [['S', 8], ['G', 5]],
        'D': [['A', 3]],
        'E': [['A', 7]]}

tree2 = {'S': [['A', 1], ['B', 2]],
         'A': [['S', 1]],
         'B': [['S', 2], ['C', 3], ['D', 4]],
         'C': [['B', 2], ['E', 5], ['F', 6]],
         'D': [['B', 4], ['G', 7]],
         'E': [['C', 5]],
         'F': [['C', 6]]
         }

heuristic = {'S': 8, 'A': 8, 'B': 4, 'C': 3, 'D': 5000, 'E': 5000, 'G': 0}
heuristic2 = {'S': 0, 'A': 5000, 'B': 2, 'C': 3, 'D': 4, 'E': 5000, 'F': 5000, 'G': 0}

cost = {'S': 0}             # total cost for nodes visited


def AStarSearch():
    global tree, heuristic
    closed = []             # closed nodes
    opened = [['S', 8]]     # opened nodes

    '''find the visited nodes'''
    while True:
        print("-------------------------------------------------------------------------------")
        print("opened= ",opened)
        print("closed= ", closed)
        fn = [i[1] for i in opened]     # fn = f(n) = g(n) + h(n)
        print("fn=",fn)
        chosen_index = fn.index(min(fn))
        print("chosen_index=",chosen_index)
        node = opened[chosen_index][0]  # current node
        print("node=", node)
        closed.append(opened[chosen_index])
        del opened[chosen_index]
        if closed[-1][0] == 'G':        # break the loop if node G has been found
            break
        for item in tree[node]:
            if item[0] in [closed_item[0] for closed_item in closed]:
                continue
            cost.update({item[0]: cost[node] + item[1]})            # add nodes to cost dictionary
            fn_node = cost[node] + heuristic[item[0]] + item[1]     # calculate f(n) of current node
            temp = [item[0], fn_node]
            opened.append(temp)                                     # store f(n) of current node in array opened

    '''find optimal sequence'''
    trace_node = 'G'                        # correct optimal tracing node, initialize as node G
    optimal_sequence = ['G']                # optimal node sequence
    for i in range(len(closed)-2, -1, -1):
        check_node = closed[i][0]           # current node
        if trace_node in [children[0] for children in tree[check_node]]:
            children_costs = [temp[1] for temp in tree[check_node]]
            children_nodes = [temp[0] for temp in tree[check_node]]

            '''check whether h(s) + g(s) = f(s). If so, append current node to optimal sequence
            change the correct optimal tracing node to current node'''
            if cost[check_node] + children_costs[children_nodes.index(trace_node)] == cost[trace_node]:
                optimal_sequence.append(check_node)
                trace_node = check_node
    optimal_sequence.reverse()              # reverse the optimal sequence

    return closed, optimal_sequence


if __name__ == '__main__':
    visited_nodes, optimal_nodes = AStarSearch()
    print('visited nodes: ' + str(visited_nodes))
    print('optimal nodes sequence: ' + str(optimal_nodes))


################################################################################

'''
Disjoint Set Data Structures

Last Updated: 07-06-2020
Consider a situation with a number of persons and following tasks to be performed on them.

Add a new friendship relation, i.e., a person x becomes friend of another person y.
Find whether individual x is a friend of individual y (direct or indirect friend)
Example:

We are given 10 individuals say,
a, b, c, d, e, f, g, h, i, j

Following are relationships to be added.
a <-> b  
b <-> d
c <-> f
c <-> i
j <-> e
g <-> j

And given queries like whether a is a friend of d
or not.

We basically need to create following 4 groups
and maintain a quickly accessible connection
among group items:
G1 = {a, b, d}
G2 = {c, f, i}
G3 = {e, g, j}
G4 = {h}

########################

Kruskal’s Minimum Spanning Tree Algorithm

A minimum spanning tree (MST) or minimum weight spanning tree for a weighted, 
connected and undirected graph is a spanning tree with weight less than or 
equal to the weight of every other spanning tree.

step 1: sort the edges in an ascending order and identify the source and destination node
step 2: keep adding the nodes and edges from the smallest one until number of edges minus one
step 3: if adding an edge creates a cycle skip it

'''

classSet = []
class DisjointSet:
    #parent is the first element of every array of the set
    def __init__(self, item):
        self.item = item
        self.rank = 0      
        classSet.append([self.item])
        
    def findWithRank(self, itm):
        #if necessary it sets the parent and returns the parent's item value
        for i in range(len(classSet)):
            if itm.item in classSet[i]:
                break
        return (classSet[i][0], itm.rank)
        
    def union(self, itm1, itm2):
        #combine two sets
        parentSet = 0
        parent1, rank1 = self.findWithRank(itm1)
        parent2, rank2 = self.findWithRank(itm2)
        if rank1 >= rank2:
            parent = parent1
            parentSet = 1
        else:
            parent = parent2
            parentSet = 2
        self.makeSet(parent, parentSet, itm1.item, itm2.item, itm1, itm2)
         
    def makeSet(self, parent, parentSet, itm1, itm2, obj1, obj2):
        #find the sets in which itm1 and itm2 belong to
        itm1SetIndex = -1
        itm2SetIndex = -1     
        newSet = []
        maxRank = 0
        for i in range(len(classSet)):
            if itm1 in classSet[i]:
                itm1SetIndex = i
            if itm2 in classSet[i]:
                itm2SetIndex = i
            if itm1SetIndex != -1 and itm2SetIndex != -1:#if indexes were found
                break
        
        if parentSet == 1:
            classSet[itm1SetIndex].remove(parent)   
            maxRank = obj1.rank + 1
        elif parentSet == 2:
            classSet[itm2SetIndex].remove(parent)
            maxRank = obj2.rank + 1
        newSet.append(parent)

        for i in classSet[itm1SetIndex]:
            newSet.append(i)
        for i in classSet[itm2SetIndex]:
            newSet.append(i)        
        
        classSet.pop(itm1SetIndex)
        if itm1SetIndex < itm2SetIndex:
            itm2SetIndex -= 1
        classSet.pop(itm2SetIndex)  
        
        classSet.append(newSet)
        
class Kruskal_MST:
    def __init__(self, graph):
        self.graph = graph
        self.DS = DisjointSet(None)
    
    def sortWeights(self):
        return sorted(self.graph, key=self.weight)
    def weight(self, touple):
        return touple[2]
        
    def isCycle(self, touple):
        for s in classSet:
            if touple[0] in s and touple[1] in s:
                return True
        return False
    
    def addToMST(self, touple):
        if not self.isCycle(touple):   
            obj1 = DisjointSet(touple[0])
            obj2 = DisjointSet(touple[1])
            self.DS.union(obj1, obj2)
    
    def run(self, numberOfVertices):
        self.graph = self.sortWeights()
        for i in range(numberOfVertices-1):
            self.addToMST(self.graph[i])
        print(self.graph)
        print(classSet)
        
#test case for Kruskal Algorithm
#visual example of the data set: https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
#, (1,2,8), (7,8,7), (2,8,2), (8,6,6), (7,6,1), (6,5,2), (2,5,4), (2,3,7), (3,5,14), (3,4,9), (5,4,10)        
krk = Kruskal_MST([(1,7,11), (0,1,4), (0,7,8)]) #union doesn't work.
krk.run(4)    
        
'''        
#test case 1 for disjoint set    
obj1 = DisjointSet(1)       
obj2 = DisjointSet(2)       
obj3 = DisjointSet(3)       
obj4 = DisjointSet(4)       
obj5 = DisjointSet(5)       
obj6 = DisjointSet(6)       
obj7 = DisjointSet(7) 

print(classSet)
                

obj1.union(obj1, obj2)
obj1.union(obj2, obj3)
obj1.union(obj4, obj5)
obj1.union(obj6, obj7)
obj1.union(obj5, obj6)
obj1.union(obj3, obj7)
    
print(classSet)


#test case 2 for disjoint set   
obj1 = DisjointSet('a')       
obj2 = DisjointSet('b')       
obj3 = DisjointSet('c')       
obj4 = DisjointSet('d')       
obj5 = DisjointSet('e')       
obj6 = DisjointSet('f')       
obj8 = DisjointSet('g')      
obj9 = DisjointSet('h')
obj10 = DisjointSet('i')
obj11 = DisjointSet('j')

print(classSet)

obj1.union(obj1, obj2)
obj1.union(obj2, obj4)
obj1.union(obj3, obj6)
obj1.union(obj3, obj10)
obj1.union(obj11, obj5)
obj1.union(obj8, obj11)

print(classSet)

'''

################################################################################

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


################################################################################

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

################################################################################

