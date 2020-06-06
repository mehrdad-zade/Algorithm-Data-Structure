#1###############################################################################

'''
this problem gives you an array with all zeros. you have to go through the queries which provides you with a range and val. the
val must be added to all the elements in that range. return the maximum value in the resulting arr.
sol: add the value to the begining of the range and substract val from the end of the range + 1. then loop through the array and
do a prefix addition (this will add the value for each of the elements within expected range. +1 for end of range is because
you only want to add values to elements in the range inclusively, so by subtracting from end+1 you will start the next range with
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