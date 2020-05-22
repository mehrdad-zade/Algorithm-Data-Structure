'''

https://www.geeksforgeeks.org/a-search-algorithm/

https://www.geeksforgeeks.org/minimum-cost-graph/?ref=leftbar-rightbar

https://www.geeksforgeeks.org/traveling-salesman-problem-using-genetic-algorithm/?ref=leftbar-rightbar

https://www.geeksforgeeks.org/count-total-ways-to-reach-destination-from-source-in-an-undirected-graph/?ref=leftbar-rightbar

https://www.geeksforgeeks.org/add-and-remove-edge-in-adjacency-list-representation-of-a-graph/?ref=leftbar-rightbar

https://www.geeksforgeeks.org/implementing-water-supply-problem-using-breadth-first-search/?ref=leftbar-rightbar

https://www.geeksforgeeks.org/minimum-cost-to-reach-from-the-top-left-to-the-bottom-right-corner-of-a-matrix/?ref=leftbar-rightbar

https://www.geeksforgeeks.org/find-the-maximum-of-all-distances-to-the-nearest-1-cell-from-any-0-cell-in-a-binary-matrix/?ref=leftbar-rightbar

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
