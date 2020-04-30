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
