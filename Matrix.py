#1##########################################################################################

'''
Matrix Multiplication : O(N^3)

+ naive sol is O(N^3)

void multiply(int A[][N], int B[][N], int C[][N]) 
{ 
    for (int i = 0; i < N; i++) 
    { 
        for (int j = 0; j < N; j++) 
        { 
            C[i][j] = 0; 
            for (int k = 0; k < N; k++) 
            { 
                C[i][j] += A[i][k]*B[k][j]; 
            } 
        } 
    } 
} 

+ Strassen's Alg is O(N^2), but it takes more space and it's inefficient for most problems.
sol: id divides the matrix into n/2 sub matrixes and applies the multiplications and additions.
instead of the treditional 8 multiplication, strassen manages to do it with 7 formulas.

'''

#2########################################################################################


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


#3########################################################################################
'''
Matrix Chain Multiplication | DP-8

Last Updated: 25-08-2020
Given a sequence of matrices, find the most efficient way to multiply these matrices together. The problem is not actually to perform the multiplications, but merely to decide in which order to perform the multiplications.

We have many options to multiply a chain of matrices because matrix multiplication is associative. In other words, no matter how we parenthesize the product, the result will be the same. For example, if we had four matrices A, B, C, and D, we would have:

    (ABC)D = (AB)(CD) = A(BCD) = ....
However, the order in which we parenthesize the product affects the number of simple arithmetic operations needed to compute the product, or the efficiency. For example, suppose A is a 10 × 30 matrix, B is a 30 × 5 matrix, and C is a 5 × 60 matrix. Then,

    (AB)C = (10×30×5) + (10×5×60) = 1500 + 3000 = 4500 operations
    A(BC) = (30×5×60) + (10×30×60) = 9000 + 18000 = 27000 operations.


 Input: p[] = {40, 20, 30, 10, 30}   
  Output: 26000  
  There are 4 matrices of dimensions 40x20, 20x30, 30x10 and 10x30.
  Let the input 4 matrices be A, B, C and D.  The minimum number of 
  multiplications are obtained by putting parenthesis in following way
  (A(BC))D --> 20*30*10 + 40*20*10 + 40*10*30    
'''

# A naive recursive implementation that 
# simply follows the above optimal  
# substructure property. there is a complex dynamic solution to it too
import sys 
  
# Matrix A[i] has dimension p[i-1] x p[i] 
# for i = 1..n 
def MatrixChainOrder(p, i, j): 
  
    if i == j: 
        return 0
  
    _min = sys.maxsize 
      
    # place parenthesis at different places  
    # between first and last matrix,  
    # recursively calculate count of 
    # multiplications for each parenthesis 
    # placement and return the minimum count 
    for k in range(i, j): 
      
        count = (MatrixChainOrder(p, i, k)  
             + MatrixChainOrder(p, k + 1, j) 
                   + p[i-1] * p[k] * p[j]) 
  
        if count < _min: 
            _min = count; 
      
  
    # Return minimum count 
    return _min; 
  
  
# Driver program to test above function 
arr = [1, 2, 3, 4, 3]; 
n = len(arr); 
  
print("Minimum number of multiplications is ", 
                MatrixChainOrder(arr, 1, n-1)); 


#4########################################################################################
'''
Maximum size square sub-matrix with all 1s:
    
Given a binary matrix, find out the maximum size square sub-matrix with all 1s.

For example, consider the below binary matrix.
M[R][C] = [         [0, 1, 1, 0, 1],  
                    [1, 1, 0, 1, 0],  
                    [0, 1, 1, 1, 0],  
                    [1, 1, 1, 1, 0],  
                    [1, 1, 1, 1, 1],  
                    [0, 0, 0, 0, 0]];  
maximum-size-square-sub-matrix-with-all-1s is 3.

solution dynamic programming:
    1 - create an auxilery matrix. it starts with all the values from 1st row and column. 
        also, all 0's from M will be added. all else will be blank.
        
        aux[R][C] = [[0, 1, 1, 0, 1],  
                     [1, _, 0, _, 0],  
                     [0, _, _, _, 0],  
                     [1, _, _, _, 0],  
                     [1, _, _, _, _],  
                     [0, 0, 0, 0, 0]];  

    2 - find the minimum of (top, top left, and buttom left) + 1, for all blank values
    
   0  1  1  0  1
   1  1  0  1  0
   0  1  1  1  0
   1  1  2  2  0
   1  2  2  3  1
   0  0  0  0  0
    
    3 - find the max value in aux
    4 - based on the position of max value you can identify the requested square
    
time complexity: O(RC)
'''
def maxSizeSquare(M):
    
    R = len(M) #number of columns
    C = len(M[0])    #number of rows
    
    aux = [[None for i in range(C)] for j in range(R)]
    
    aux[0] = M[0]
    
    for i in range(R):
        aux[i][0] = M[i][0]
        
    for i in range(1, R):
        for j in range(1, C):            
            if M[i][j] == 0:
                aux[i][j] = 0
            
            
    for i in range(1, R):
        for j in range(1, C):            
            if M[i][j] == 1:
                aux[i][j] = min(aux[i][j-1], aux[i-1][j], aux[i-1][j-1]) + 1
            else:
                aux[i][j] = 0
                
    _max = 0
    for i in range(R):
        for j in range(C): 
            if aux[i][j]  > _max:
                _max = aux[i][j]
                
    
        
    return(_max)
        
        
M =          [[0, 1, 1, 0, 1],  
                    [1, 1, 0, 1, 0],  
                    [0, 1, 1, 1, 0],  
                    [1, 1, 1, 1, 0],  
                    [1, 1, 1, 1, 1],  
                    [0, 0, 0, 0, 0]]
            
maxSizeSquare(M)            


