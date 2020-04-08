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
