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

################################################################################

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

################################################################################
