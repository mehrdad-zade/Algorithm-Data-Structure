######################################################################################
'''
you cannot create a set from list of lists because they are not hashable
you need to convert  each row in the list to a tuple and find the set


>>> mat = [[1,2,3],[4,5,6],[1,2,3],[7,8,9],[4,5,6]]
>>> set(tuple(row) for row in mat)
set([(4, 5, 6), (7, 8, 9), (1, 2, 3)])
'''

#1###############################################################################

'''
this problem gives you an array with all zeros. you have to go through 
the queries which provides you with a range and val. the
val must be added to all the elements in that range. return the maximum 
value in the resulting arr.

sol: add the value to the begining of the range and substract val from 
the end of the range + 1. then loop through the array and
do a prefix addition (this will add the value for each of the elements 
ithin expected range. +1 for end of range is because
you only want to add values to elements in the range inclusively, so by 
ubtracting from end+1 you will start the next range with
prefix addition without considering the value from previous range)
'''
def maxRangeAdditions(n, queries):
    arr = [0 for i in range(n)]
    queries_Rows = 0
    while queries_Rows < len(queries) :
        arr[queries[queries_Rows][0] - 1] += queries[queries_Rows][2]
        if queries[queries_Rows][1] != n :
            arr[queries[queries_Rows][1]] -= queries[queries_Rows][2]
        queries_Rows += 1

    maxVal = arr[0]
    for i in range(1,n,1) :
        arr[i] += arr[i-1]
        if arr[i] > maxVal :
            maxVal = arr[i]
    return maxVal

n = 9
queries = [[1, 5, 3],
           [4, 8, 7],
           [6, 9, 1]]
print("array addition, max occurence with added num: ", maxRangeAdditions(n, queries))

#2#######################################################################################################################

'''
find all the subarrays
'''

def subArrays(arr):
    sub = [[]]
    for i in range(len(arr)+1):
        for j in range(i, len(arr)+1):
            temp = arr[i:j]
            if temp:
                sub.append(temp)
    return sub

arr = [1, 2, 3] 
print("sub-arrays = ", subArrays(arr)) 

#find all sub strings
subs = [s[i: j] for i in range(n) for j in range(i + 1, n + 1)]

#3###############################################################################

'''
find unique premutations of a set
'''
from itertools import permutations
x = [1,1,0,0]
perm = set(permutations(x))

print(perm)



# withouy librery
def permutations(s):
    if len(s) <= 1:
        return [s]
    else:
        perms = []
        for e in permutations(s[:-1]):
            for i in range(len(e)+1):
                perms.append(e[:i] + [s[-1]] + e[i:])
    return perms

print(permutations([1,1,0,0])) # if you want unique ones, make it a set

#4###############################################################################

'''
Find whether an array is subset of another array.

sol: sort both arrays and compare one by one. or, sort first arr and bin search elements of second arr
'''

def isSubArr(arr1, arr2):#find out if arr2 is a sub-array of arr1
    n1 = len(arr1)
    n2 = len(arr2)
    if n2 > n1: # if arr2 is bigger, then it cannot be the sub-array of arr1
        return False
    arr1.sort()
    arr2.sort()
    j = 0
    for i in range(n1):
        print("i = ", i)
        print("j = ", j)
        if arr1[i] == arr2[j]:
            j += 1
        elif arr1[i] > arr2[j]:
            return False
        else: # if arr1<arr2
            continue
    if j != n2:
        return False
    return True

arr1 = [11, 1, 13, 21, 3, 7]; 
arr2 = [11, 1, 13, 21, 7]; 
print("is arr2 a subset of arr1? ", isSubArr(arr1, arr2))

#5###############################################################################
'''
rotate elements of the array d times. if d=2 and array is 1,2,3,4; then one time rotation would result
in: arry = 3,4,1,2.
suppose d is static and you will apply it many times. write an efficient program to rotate
the array eleemnts.

class Rotator:
    def __init__(self, arr, d):
        self.arr = arr
        self.d = d % len(arr)  # Normalize d in case it's larger than the array length

    def rotate(self):
        """Rotates the array in O(n) time and O(1) space"""
        self.arr = self.arr[self.d:] + self.arr[:self.d]

    def get_array(self):
        return self.arr

# Example Usage
arr = [1, 2, 3, 4]
rotator = Rotator(arr, 2)
rotator.rotate()
print(rotator.get_array())  # Output: [3, 4, 1, 2]



#6#############################################################################
'''
Given a value N, if we want to make change for N cents, and we have infinite 
supply of each of S = { S1, S2, .. , Sm} valued coins, how many ways can we 
make the change? The order of coins doesn’t matter.

For example, for N = 4 and S = {1,2,3}, there are four solutions: {1,1,1,1},
{1,1,2},{2,2},{1,3}. So output should be 4. For N = 10 and S = {2, 5, 3, 6}, 
there are five solutions: {2,2,2,2,2}, {2,2,3,3}, {2,2,6}, {2,3,5} and {5,5}. 
So the output should be 5.

you can see the recursive solution for better understanding the way to solve this, but
we can do it with dynamic programming to save time O(mn)

'''

# Returns the count of ways we can sum 
# S[0...m-1] coins to get sum n 
def count1(S, m, n ): 
  
    # If n is 0 then there is 1 
    # solution (do not include any coin) 
    if (n == 0): 
        return 1
  
    # If n is less than 0 then no 
    # solution exists 
    if (n < 0): 
        return 0; 
  
    # If there are no coins and n 
    # is greater than 0, then no 
    # solution exist 
    if (m <=0 and n >= 1): 
        return 0
  
    # count is sum of solutions (i)  
    # including S[m-1] (ii) excluding S[m-1] 
    return count1( S, m - 1, n ) + count1( S, m, n-S[m-1] );

def count2(S, m, n):
    # We need n+1 rows as the table is constructed 
    # in bottom up manner using the base case 0 value
    # case (n = 0)
    table = [[0 for x in range(m)] for x in range(n+1)]
  
    # Fill the entries for 0 value case (n = 0)
    for i in range(m):
        table[0][i] = 1
  
    # Fill rest of the table entries in bottom up manner
    for i in range(1, n+1):
        for j in range(m):
  
            # Count of solutions including S[j]
            x = table[i - S[j]][j] if i-S[j] >= 0 else 0
  
            # Count of solutions excluding S[j]
            y = table[i][j-1] if j >= 1 else 0
  
            # total count
            table[i][j] = x + y
  
    return table[n][m-1]
    
    
arr = [1, 2, 3] 
m = len(arr) 
n = 4
print(count2(arr, m, n))


#7########################################################################################

"""
Naive algorithm for Pattern Searching
O(m*n)

have a window of size len of pat on the txt. move one by one and if the entire
window is a match, add the starting point idx to a list.
"""

def NaiveSearch(pat, txt): 
    if len(pat) > len(txt):
        return []
    txt_list = []
    pat_list = []
    idx_list = []
    for ch in txt:
        txt_list.append(ch)
    for ch in pat:
        pat_list.append(ch)
    n = len(txt_list)
    m = len(pat_list)
    
    idx = 0
    while idx + m <= n:
        for j in range(m):
            if txt_list[idx+j] != pat_list[j]:#if the indexes are not having same value then pat is not within this windows of txt
                break
        if j == m-1:#if above "if" statement didn't break till the end of pat chars then entire window must have been a match
            idx_list.append(idx)
        idx += 1
    return idx_list
            
print(NaiveSearch("AABA", "AABAACAADAABAABA"))


#8########################################################################################

"""
KML Algorithm for pattern matching.
O(n+m)

https://www.youtube.com/watch?v=GTJr8OvyEVQ
"""

def KMPSearch(pat, txt): 
    M = len(pat) 
    N = len(txt) 
  
    # create repetition[] that will hold the prefix suffix indexes for pattern 
    repetition = [0]*M 
    
    # Preprocess the pattern (calculate repetition[] array) 
    repetition = computeRepetitionArray(pat, M, repetition) 
  
    i = 0 # index for txt[] 
    j = 0 # index for pat[] 
    while i < N: 
        if pat[j] == txt[i]: 
            i += 1
            j += 1
  
        if j == M: 
            print ("Found pattern at index " + str(i-j)) 
            j = repetition[j-1] 
  
        # mismatch after j matches 
        elif i < N and pat[j] != txt[i]: 
            # Do not match repetition[0..repetition[j-1]] characters, 
            # they will match anyway 
            if j != 0: 
                j = repetition[j-1] 
            else: 
                i += 1
  
def computeRepetitionArray(pat, M, repetition): 
    j = 0 # length of the previous longest prefix suffix 
  
    repetition[0] # repetition[0] is always 0 
    i = 1
  
    # the loop calculates repetition[i] for i = 1 to M-1 
    while i < M: 
        if pat[i] == pat[j]: 
            j += 1
            repetition[i] = j
            i += 1
        else: 
            if j != 0: #if not equal and j not zero
                j = repetition[j-1] 
            else: #if they are not equal
                repetition[i] = 0
                i += 1
    return repetition
  
txt = "ABABDABACDABABCABAB"
pat = "ABABCABAB"
KMPSearch(pat, txt) 
  

#9########################################################################################
'''
Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

 

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

'''


from itertools import chain, combinations
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        return list(chain.from_iterable(combinations(nums, i) for i in range(len(nums)+1)))


#10########################################################################################
