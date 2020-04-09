#-----------------------------------------Arrays--------------------------------------------------

def minSwaps(arr):
    n = len(arr)

    # Create two arrays and use
    # as pairs where first array
    # is element and second array
    # is position of first element
    arrpos = [*enumerate(arr)]

    # Sort the array by array element
    # values to get right position of
    # every element as the elements
    # of second array.
    arrpos.sort(key = lambda it:it[1])

    # To keep track of visited elements.
    # Initialize all elements as not
    # visited or false.
    vis = {k:False for k in range(n)}

    # Initialize result
    ans = 0
    print(arrpos)
    for i in range(n):
        print("i :",i)
        print("arrpos[i][0] : ", arrpos[i][0])
        # alreadt swapped or
        # alreadt present at
        # correct position
        if vis[i] or arrpos[i][0] == i:
            continue

        # find number of nodes
        # in this cycle and
        # add it to ans

        cycle_size = 0
        j = i
        while not vis[j]:

            # mark node as visited
            vis[j] = True

            # move to next node
            j = arrpos[j][0]
            cycle_size += 1

        # update answer by adding
        # current cycle
            print("arr[j]: ", arr[j])
            print("vis :", vis)

        if cycle_size > 0:
            ans += (cycle_size - 1)

    # return answer
    return ans

# Driver Code
arr = [4,1,3,2]
print(minSwaps(arr))


'''
this problem gives you an array with all zeros. you have to go through the queries which provides you with a range and val. the
val must be added to all the elements in that range and return the maximum value in the resulting arr.
sol: add the value to the begining of the range and substract val from the end of the range + 1. then loop through the array and
do a prefix addition (this will add the value for each of the elements within expected range. +1 for end of range is because
you only want to add values to elements in the range inclusively, so by subtracting from end+1 you will start the next range with
prefix addition without considering the value from previous range)
'''
def arrayManipulation(n, queries):
    arr = [0 for i in range(n)]
    qRow = 0
    while qRow < len(queries) :
        arr[queries[qRow][0] - 1] += queries[qRow][2]
        if queries[qRow][1] != n :
            arr[queries[qRow][1]] -= queries[qRow][2]
        qRow += 1

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
print("array addition, max occurence with added num: ", arrayManipulation(n, queries))


#---------------------------------Dictionaries----------------------------------------------------------------------


'''
Given a string of lower alphabet characters, count total substring of this string which are anagram to each other.
Input  : str = “xyyx”
Output : 4
{“x”, “x”}, {"y", "y"}, {“xy”, “yx”}, {“xyy”, “yyx”}
'''
def countOfAnagramSubstring(s):

    # Returns total number of anagram
    # substrings in s
    n = len(s)
    mp = dict()

    # loop for length of substring
    for i in range(n):
        sb = ''
        for j in range(i, n):
            sb = ''.join(sorted(sb + s[j])) #from j to end keep adding the sub strings, sorted
            print (sb)
            mp[sb] = mp.get(sb, 0) #get the value of key sb, if doesn't exist return 0

            # increase count corresponding
            # to this dict array
            mp[sb] += 1 # ++ the value for the key. key is one of the possible substrings
    print (mp)
    anas = 0

    # loop over all different dictionary
    # items and aggregate substring count
    for k, v in mp.items():
        anas += (v*(v-1))//2 #formula for the toal anagrams
    return anas

# Driver Code
s = "xyyx"
print(countOfAnagramSubstring(s))
