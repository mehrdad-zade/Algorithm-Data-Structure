'''
a sorting algorithm with O(n) is not algorithmically possible. however, since the best algorithm
will take O(N lgN) time, if we have lgN number of processors, then we can complete the sorting
in O(N)
'''


################################################################################

'''
find the k's largest/smallest val
https://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array/
'''


#sort a touple of touples based on the first elements
def sortElementWise(touples):
    #first is a function
    #sorted is an internal function
  return sorted(touples, key=first)

def first(touple):
    #you can choose to sort based on other elements too
  return touple[0]

#Test Case
touples = ((2,3), (4,1), (1,9), (9,0), (5,7))
print(sortElementWise(touples))

################################################################################

#sort a touple based on 1st AND 2nd element
def sort(touples):
  return sorted(touples, key=lambda touples: (touples[0], touples[1]))

#Test Case
touples = ((2,3), (4,1), (1,9), (9,0), (1,4), (2,2), (2,1), (5,7))
print(sort(touples))

################################################################################
'''
Given two sorted arrays, the task is to merge them in a sorted manner.
'''
def mergeArrays(arr1, arr2): 
    n = len(arr1)
    m = len(arr2)
    arr3 = []
    i, j = 0, 0

  
    # Traverse both array 
    while i < n and j < m: 
        if arr1[i] < arr2[j]: 
            arr3.append(arr1[i])
            i = i + 1
        else: 
            arr3.append(arr2[j])
            j = j + 1
    # Store remaining elements 
    # of first array 
    while i < n: 
        arr3.append(arr1[i])
        i = i + 1 
    # Store remaining elements  
    # of second array 
    while j < m: 
        arr3.append(arr2[j])
        j = j + 1
    print("Array after merging", arr3) 

        
#Test Case
arr1 = [5,2,3,7,1]
arr2 = [10, 1, 2, 9, 0]
arr1.sort()
arr2.sort()
mergeArrays(arr1, arr2)

################################################################################
'''
merge sort: divide the arr like a bin tree up to a point where arr has 1 element.
then start merging the elements back-up into the tree to sort the entire arr.

time  : O(n lgn)
space : O(n)

below link has a good dipiction 

https://en.wikipedia.org/wiki/File:Merge_sort_algorithm_diagram.svg
'''
def mergeSort(arr):
    n = len(arr)
    if n == 1:
        return arr
    m = int(n/2)
    l = mergeSort(arr[:m])
    r = mergeSort(arr[m:])    
    return merger(l, r)

def merger(l, r):
    len_l = len(l)
    len_r = len(r)
    j = 0
    i = 0
    temp = []
    while i < len_l and j < len_r:
        if l[i] <= r[j]:
            temp.append(l[i])
            i += 1
        else:
            temp.append(r[j])
            j += 1
    while i < len_l:
        temp.append(l[i])
        i += 1
    while j < len_r:
        temp.append(r[j])
        j += 1
        
    return temp

#test case
arr = [38, 27, 43, 3, 9, 82, 10]
print("Merge Sort Result : ", mergeSort(arr))

################################################################################
'''
heap sort

time  : O(n lgn)
space : O(1)

you have to manage an array based on a full tree concept. the tree has to be ordered such that the parent of each node is bigger than its children.
if the child is smaller you'll have to swap on the tree and then swap on the array too. once there is nothing else to swap then
swap the root with the last element. last element will be dropped from the tree and from the array since it's at the end of the 
array we should keep tracker to know which item is the last element on the array which hasn't already been sorted.
this way you'll get a ascending array
'''

def heapSort(arr):
    n = len(arr)
    #max heap
    for i in range(n,-1,-1):
        heapify(arr, i, n)
    
    #keep largest at the end
    for i in range(n-1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, 0, i)
    
    return arr


    
def heapify(arr, i, n):    
    largest_index = i
    l = 2 * i + 1
    r = 2 * i + 2
    if l < n and arr[largest_index] < arr[l]:
        largest_index = l
    if r < n and arr[largest_index] < arr[r]:
        largest_index = r
    if largest_index != i:
        arr[largest_index], arr[i] = arr[i], arr[largest_index]
        heapify(arr, largest_index, n)
    
    
#test cases       
arr = [9,2,5,3,8,10,6,5,4,7]
print(heapSort(arr))    

################################################################################
'''
Counting Sort:
Counting sort is a sorting technique based on keys between a specific range. 

elements are in the range 1 to k

time  : O(n+k)
space : O(k)

https://www.youtube.com/watch?v=7zuGmKfUt7s
'''

def countSort(arr, k):
    n = len(arr)
    index = [0 for i in range(k)]
    output = [-1 for i in range(n+1)]
    #counting reoccurences at different indexes
    for e in arr:
        index[e] += 1
    
    #sum elements at each index with prev one
    for i in range(1,k,1):
        index[i] += index[i-1]

    #loop through arr, the indx of e in arr is in index which can be assigned to output
    for i in range(n):
        output[index[arr[i]]] = arr[i]
        index[arr[i]] -= 1

    return output[1:]

arr = [1,4,1,2,7,5,2]
print("Count Sort output : ", countSort(arr, 9))

################################################################################
'''
Radix Sort
What if the elements are in range from 1 to n^2? 
We can’t use counting sort because counting sort will take O(n^2).
The idea of Radix Sort is to do digit by digit sort starting from least 
significant digit to most significant digit. Radix sort uses counting 
sort as a subroutine to sort. 

time  : O(nk)
space : O(n+k)
'''

def radixSort(arr):
    n = len(arr)
    maxVal = max(arr)
    exp = 1
    while maxVal/exp > 0:
        arr = countingSort_util(arr, exp, 10, n)
        exp *= 10
    return arr
        
def countingSort_util(arr, exp, k, n):
    count = [0 for i in range(k)]
    output = [0 for i in range(n)]

    for e in arr:
        indx = int(e / exp)
        count[indx%k] += 1

    for i in range(1,k):
        count[i] += count[i-1]
        
    i = n - 1
    while i >= 0:
        indx = int(arr[i]/exp)
        output[count[indx%k] - 1] = arr[i]
        count[indx%k] -= 1
        i -= 1
    return output

arr = [170, 45, 75, 90, 802, 24, 2, 66]
print("Count Sort output : ", radixSort(arr))

################################################################################
'''
Interpolation Search:
it's better than binary search if array is sorted and uniformly distributed.
average O(log(logn)); worst O(n)
'''

def interpolationSearch(arr, key):
    end = len(arr) - 1
    begin = 0
    
    while arr[begin] != arr[end] and key >= arr[begin] and key <= arr[end]:
        mid = int(begin + ((key - arr[begin]) * (end - begin) / (arr[end] - arr[begin])))
        
        if key > arr[mid]:
            begin = mid + 1
        elif key < arr[mid]:
            end = mid -1
        else:
            return "Found"
    if key == arr[begin]:
        return "Found"
    else:
        return "Not Found"
    

arr = [2, 4, 5, 6, 8, 10, 11, 15, 16]
print("Does key exisits in arr? ", interpolationSearch(arr, 10))
################################################################################
'''
Search in an almost sorted array?
Basically the element arr[i] can only be swapped with either 
arr[i+1] or arr[i-1] to get a fully sorted array. or example 
consider the array {2, 3, 10, 4, 40}

A simple solution is to linearly search the given key in given array. 
Time complexity of this solution is O(n). We can modify binary search 
to do it in O(Logn) time.

The idea is to compare the key with middle 3 elements, if present then 
return the index. If not present, then compare the key with middle element 
to decide whether to go in left half or right half.
'''           
















