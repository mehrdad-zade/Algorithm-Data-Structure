class Graph :
    def __init__(self, graph) :
        self.graph = graph
        self.numOfVertices = len(graph)
        
    def dijkstra(self, startingVertice) :
        maxValue = 1000
        shortestPathArray = [maxValue] * self.numOfVertices
        visitedVertices = [False] * self.numOfVertices
        
        shortestPathArray[startingVertice] = 0#starting point has weight=0
        
        for loop in range(self.numOfVertices) :
        
            minValue = maxValue
            minValueIndex = -1
            for v in range(self.numOfVertices) :
                if shortestPathArray[v] < minValue and visitedVertices[v] == False :
                    minvalue = shortestPathArray[v]
                    minValueIndex = v
                    
            visitedVertices[minValueIndex] = True
            
            
            for v in range(self.numOfVertices) :
                if self.graph[minValueIndex][v] > 0 and visitedVertices[v] == False and shortestPathArray[v] > shortestPathArray[minValueIndex] + self.graph[minValueIndex][v] :
                    shortestPathArray[v] = shortestPathArray[minValueIndex] + self.graph[minValueIndex][v]
            
        print(shortestPathArray)
        
        
graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0], 
        [4, 0, 8, 0, 0, 0, 0, 11, 0], 
        [0, 8, 0, 7, 0, 4, 0, 0, 2], 
        [0, 0, 7, 0, 9, 14, 0, 0, 0], 
        [0, 0, 0, 9, 0, 10, 0, 0, 0], 
        [0, 0, 4, 14, 10, 0, 2, 0, 0], 
        [0, 0, 0, 0, 0, 2, 0, 1, 6], 
        [8, 11, 0, 0, 0, 0, 1, 0, 7], 
        [0, 0, 2, 0, 0, 0, 6, 7, 0] 
        ]; 

g = Graph(graph)  
g.dijkstra(0)        