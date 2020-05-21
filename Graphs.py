'''

https://www.geeksforgeeks.org/a-search-algorithm/
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

