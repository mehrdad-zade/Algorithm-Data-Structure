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

#########################################################
