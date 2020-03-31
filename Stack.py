https://www.geeksforgeeks.org/stack-set-2-infix-to-postfix/

https://www.geeksforgeeks.org/check-for-balanced-parentheses-in-an-expression/

https://www.geeksforgeeks.org/next-greater-element/

https://www.geeksforgeeks.org/sorting-array-using-stacks/

def HanoiTower(n, Tower1, Tower2, Tower3):
  if n==1:
    print ("Move disk",n,"from tower",Tower1,"to tower",Tower2)
    return
  HanoiTower(n-1, Tower1, Tower3, Tower2)
  print ("Move disk",n,"from tower",Tower1,"to tower",Tower2)
  HanoiTower(n-1, Tower3, Tower2, Tower1)



#------------------HanoiTower---------------------------------
n=3
HanoiTower(n, 'A', 'B', 'C')
