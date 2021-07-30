# 1$ ##########################################################################################
'''
Dijkstra: returns distance from starting point for all nodes
    - have all nodes as unvisited
    - have a list of nodes representing min distance from starting point
    - from starting node find the distance to all possible immediate nodes
    - mark current node as visited
    - update the distance based on the sum of distance from the starting point
    - then go to the node with smallest distance and repeat

    https://www.youtube.com/watch?v=_lHSawdgXpI
    
'''    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:25:18 2021

@author: mehrdadalemzadeh
"""
inf = float('inf')
def dijkstra(graph, node):
    n = len(graph)
    visited = [False] * n
    min_cost = [inf] * n # initial min cost for all nodes is infinity
    min_cost[node] = 0 # the starting node's cost is zero
    
    while False in visited: # loop through untill all nodes are visited (connected graph) or the same min cost node is keep being chosen which suggest that some nodes are not reachable
        visited[node] = True # once the distance from a node is being calculated mark it as visited
        for v in range(n): # for all v's connected to current node, find if v is not visited and distance is lower than current cost update min_cost
            if graph[node][v] != 0 and visited[v] == False and min_cost[node] + graph[node][v] < min_cost[v]:
                min_cost[v] = min_cost[node] + graph[node][v]
        
        #get the next node to process. if it's the same as previous node then some nodes are not reachable or cost cannot go lower
        temp = minCostUnvisited(min_cost, visited, n)
        if temp == -1 or temp == node:
            break
        node = temp
        print(min_cost)
        
    return min_cost

def minCostUnvisited(min_cost, visited, n):
    min_node = -1
    
    #find the firsunvisited node and asume it has the min cost
    for v in range(n):
        if visited[v] == False:
            min_node = v
            break
    #find the min unvisted node
    for v in range(n):
        if visited[v] == False and min_cost[v] < min_cost[min_node]:
            min_node = v
    
    return min_node

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

#link to the depiction of above graph: https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/            
    
print(dijkstra(graph, 0))            

#2###############################################################################

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

#3###############################################################################
'''
given n number of cities and the distances between them. find k cities where k data centres
can be placed in them, while the distance from other cities are minimized.

start from any city, and choose its counterpart as the farthest city. then calculate the min
distance of all other cities from these two.

repeat the same for all other cities and find the min distance.
'''

#4###############################################################################
'''
Bellman–Ford Algorithm

Given a graph and a source vertex src in graph, find shortest paths from src 
to all vertices in the given graph. Dijkstra’s algorithm is a Greedy algorithm 
and time complexity is O(VLogV) (with the use of Fibonacci heap). 

Dijkstra doesn’t work for Graphs with negative weight edges, Bellman-Ford works 
for such graphs. Bellman-Ford is also simpler than Dijkstra and suites well for 
distributed systems. But time complexity of Bellman-Ford is O(VE), which is 
more than Dijkstra.

Solution - Dynamic programming:
1) This step initializes distances from the source to all vertices as infinite 
    and distance to the source itself as 0. Create an array dist[] of size |V|.
    
2) Do following |V|-1 times where |V| is the number of vertices in given graph.
    a) Do following for each edge u-v
        If dist[v] > dist[u] + weight of edge uv, then update dist[v]
            dist[v] = dist[u] + weight of edge uv    
            
3) This step reports if there is a negative weight cycle in graph. Do following 
    for each edge u-v
        If dist[v] > dist[u] + weight of edge uv, 
            then “Graph contains negative weight cycle”
The idea of step 3 is, step 2 guarantees the shortest distances if the graph 
doesn’t contain a negative weight cycle. If we iterate through all edges one 
more time and get a shorter path for any vertex, then there is a negative weight 
cycle.

https://www.youtube.com/watch?v=lyw4FaxrwHg
    

'''
      
class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
        
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])
    
    def printAttr(self, dist, src):
        print("Vertex Distance from Source ", src)  
        print("\nVertex\tDistance ")  
        for i in range(self.V):  
            print("{0}\t{1}".format(i, dist[i]))  
            
    def BellmanFord(self, src):
        dist = [float('Inf')] * self.V
        dist[src] = 0
        
        for _ in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float('Inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    
        for u, v, w in self.graph:
                if dist[u] != float('Inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    
        self.printAttr(dist, src)
        
g = Graph(5)  
g.addEdge(0, 1, -1)  
g.addEdge(0, 2, 4)  
g.addEdge(1, 2, 3)  
g.addEdge(1, 3, 2)  
g.addEdge(1, 4, 2)  
g.addEdge(3, 2, 5)  
g.addEdge(3, 1, 1)  
g.addEdge(4, 3, -3)  
  
# Print the solution  
g.BellmanFord(0)         


#5########################################################################################

'''
Floyd Warshall Algorithm | 

The Floyd Warshall Algorithm is for solving the All Pairs Shortest Path problem. 
The problem is to find shortest distances between every pair of vertices in a given 
edge weighted directed Graph.

Example:

Input:
       graph[][] = { {0,   5,  INF, 10},
                    {INF,  0,  3,  INF},
                    {INF, INF, 0,   1},
                    {INF, INF, INF, 0} }
which represents the following graph
             10
       (0)------->(3)
        |         /|\
      5 |          |
        |          | 1
       \|/         |
       (1)------->(2)
            3       
Note that the value of graph[i][j] is 0 if i is equal to j 
And graph[i][j] is INF (infinite) if there is no edge from vertex i to j.

Output:
Shortest distance matrix
      0      5      8      9
    INF      0      3      4
    INF    INF      0      1
    INF    INF    INF      0

solution: it builds a graph, fw_result, from the main graph. the pathes will be updated here.
we go through every possible path, and assume the path is a way for two V's to connect. if that
was the case, we'll compare the current cost fw_result[i][j], with the fw_result[i][k] + fw_result[k][j]
which ever was smaller will be the new cost.

'''
import sys
def floydWarshall(graph, n): 
    fw_result = graph
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if fw_result[i][j] is None:
                    base = sys.maxsize
                else:
                    base = fw_result[i][j]
                if fw_result[i][k] is None:
                    midPath1 = sys.maxsize
                else:
                    midPath1 = fw_result[i][k]                    
                if fw_result[k][j] is None:
                    midPath2 = sys.maxsize
                else:
                    midPath2 = fw_result[k][j]
                fw_result[i][j] = min(base, midPath1 + midPath2 )
                if fw_result[i][j] > 999:
                    fw_result[i][j] = None
    return fw_result
                
#test case    
graph = [[0,5,None,10], 
             [None,0,3,None], 
             [None, None, 0,   1], 
             [None, None, None, 0] 
        ] 
print(floydWarshall(graph, len(graph)))


