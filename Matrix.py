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