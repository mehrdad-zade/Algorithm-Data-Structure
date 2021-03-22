#1##########################################################################################

'''
Closest Pair of Points | O(nlogn)

We are given an array of n points in the plane, and the problem is to find out the closest pair of points in the array.
||xy|| = sqrt((xi-xj)^2 + (yi-yj)^2)

sol:
1) We sort all points according to x coordinates.

2) Divide all points in two halves.

3) Recursively find the smallest distances in both subarrays.

4) Take the minimum of two smallest distances. Let the minimum be d.

5) Create an array strip[] that stores all points which are at most d distance away from the middle line dividing the two sets.

6) Find the smallest distance in strip[].

7) Return the minimum of d and the smallest distance calculated in above step 6.
'''


    for i in range(min(6, points_counts - 1), points_counts):
        for j in range(max(0, i - 6), i):
            current_dis = euclidean_distance_sqr(points[i], points[j])
            if current_dis < min_dis:
                min_dis = current_dis
    return min_dis


def closest_pair_of_points_sqr(points_sorted_on_x, points_sorted_on_y, points_counts):
    """divide and conquer approach
    Parameters :
    points, points_count (list(tuple(int, int)), int)
    Returns :
    (float):  distance btw closest pair of points
    >>> closest_pair_of_points_sqr([(1, 2), (3, 4)], [(5, 6), (7, 8)], 2)
    8
    """

    # base case
    if points_counts <= 3:
        return dis_between_closest_pair(points_sorted_on_x, points_counts)

    # recursion
    mid = points_counts // 2
    closest_in_left = closest_pair_of_points_sqr(
        points_sorted_on_x, points_sorted_on_y[:mid], mid
    )
    closest_in_right = closest_pair_of_points_sqr(
        points_sorted_on_y, points_sorted_on_y[mid:], points_counts - mid
    )
    closest_pair_dis = min(closest_in_left, closest_in_right)

    """
    cross_strip contains the points, whose Xcoords are at a
    distance(< closest_pair_dis) from mid's Xcoord
    """

    cross_strip = []
    for point in points_sorted_on_x:
        if abs(point[0] - points_sorted_on_x[mid][0]) < closest_pair_dis:
            cross_strip.append(point)

    closest_in_strip = dis_between_closest_in_strip(
        cross_strip, len(cross_strip), closest_pair_dis
    )
    return min(closest_pair_dis, closest_in_strip)


def closest_pair_of_points(points, points_counts):
    """
    >>> closest_pair_of_points([(2, 3), (12, 30)], len([(2, 3), (12, 30)]))
    28.792360097775937
    """
    points_sorted_on_x = column_based_sort(points, column=0)
    points_sorted_on_y = column_based_sort(points, column=1)
    return (
        closest_pair_of_points_sqr(
            points_sorted_on_x, points_sorted_on_y, points_counts
        )
    ) ** 0.5


if __name__ == "__main__":
    points = [(2, 3), (12, 30), (40, 50), (5, 1), (12, 10), (3, 4)]
    print("Distance:", closest_pair_of_points(points, len(points)))

#2##################################################################################################################
'''

Given two line segments (p1, q1) and (p2, q2), find if the given line segments intersect with each other.

sol:
Before we discuss solution, let us define notion of orientation. Orientation of an ordered triplet of points in the plane can be
–counterclockwise
–clockwise
–colinear

How is Orientation useful here?
Two segments (p1,q1) and (p2,q2) intersect if and only if one of the following two conditions is verified

1. General Case:
– (p1, q1, p2) and (p1, q1, q2) have different orientations and
– (p2, q2, p1) and (p2, q2, q1) have different orientations.

2. Special Case 
– (p1, q1, p2), (p1, q1, q2), (p2, q2, p1), and (p2, q2, q1) are all collinear and
– the x-projections of (p1, q1) and (p2, q2) intersect
– the y-projections of (p1, q1) and (p2, q2) intersect
'''
  
class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
# Given three colinear points p, q, r, the function checks if  
# point q lies on line segment 'pr'  
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
          
        # Clockwise orientation 
        return 1
    elif (val < 0): 
          
        # Counterclockwise orientation 
        return 2
    else: 
          
        # Colinear orientation 
        return 0
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def doIntersect(p1,q1,p2,q2): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)): 
        return True
  
    # Special Cases 
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
  
    # If none of the cases 
    return False
  
# Driver program to test above functions: 
p1 = Point(1, 1) 
q1 = Point(10, 1) 
p2 = Point(1, 2) 
q2 = Point(10, 2) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
  
p1 = Point(10, 0) 
q1 = Point(0, 10) 
p2 = Point(0, 0) 
q2 = Point(10,10) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
  
p1 = Point(-5,-5) 
q1 = Point(0, 0) 
p2 = Point(1, 1) 
q2 = Point(10, 10) 
  
if doIntersect(p1, q1, p2, q2): 
    print("Yes") 
else: 
    print("No") 
      

#3##################################################################################################################
'''
check if a given point lies inside or outside a polygon?

1) Draw a horizontal line to the right of each point and extend it to infinity

1) Count the number of times the line intersects with polygon edges.

2) A point is inside the polygon if either count of intersections is odd or
   point lies on an edge of polygon.  If none of the conditions is true, then 
   point lies outside.

 How to handle point ‘g’ in the above figure? 
Note that we should return true if the point lies on the line or same as one of the vertices of the given polygon. 
To handle this, after checking if the line from ‘p’ to extreme intersects, we check whether ‘p’ is colinear with 
vertices of current line of polygon. If it is coliear, then we check if the point ‘p’ lies on current side of polygon, 
if it lies, we return true, else false.
'''

# Define Infinite (Using INT_MAX  
# caused overflow problems)
INT_MAX = 10000
 
# Given three colinear points p, q, r,  
# the function checks if point q lies 
# on line segment 'pr' 
def onSegment(p:tuple, q:tuple, r:tuple) -> bool:
     
    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True
         
    return False
 
# To find orientation of ordered triplet (p, q, r). 
# The function returns following values 
# 0 --> p, q and r are colinear 
# 1 --> Clockwise 
# 2 --> Counterclockwise 
def orientation(p:tuple, q:tuple, r:tuple) -> int:
     
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))
            
    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock
 
def doIntersect(p1, q1, p2, q2):
     
    # Find the four orientations needed for  
    # general and special cases 
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if (o1 != o2) and (o3 != o4):
        return True
     
    # Special Cases 
    # p1, q1 and p2 are colinear and 
    # p2 lies on segment p1q1 
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True
 
    # p1, q1 and p2 are colinear and 
    # q2 lies on segment p1q1 
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True
 
    # p2, q2 and p1 are colinear and 
    # p1 lies on segment p2q2 
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True
 
    # p2, q2 and q1 are colinear and 
    # q1 lies on segment p2q2 
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True
 
    return False
 
# Returns true if the point p lies  
# inside the polygon[] with n vertices 
def is_inside_polygon(points:list, p:tuple) -> bool:
     
    n = len(points)
     
    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False
         
    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0
     
    while True:
        next = (i + 1) % n
         
        # Check if the line segment from 'p' to  
        # 'extreme' intersects with the line  
        # segment from 'polygon[i]' to 'polygon[next]' 
        if (doIntersect(points[i],
                        points[next], 
                        p, extreme)):
                             
            # If the point 'p' is colinear with line  
            # segment 'i-next', then check if it lies  
            # on segment. If it lies, return true, otherwise false 
            if orientation(points[i], p, 
                           points[next]) == 0:
                return onSegment(points[i], p, 
                                 points[next])
                                  
            count += 1
             
        i = next
         
        if (i == 0):
            break
         
    # Return true if count is odd, false otherwise 
    return (count % 2 == 1)
 
# Driver code
if __name__ == '__main__':
     
    polygon1 = [ (0, 0), (10, 0), (10, 10), (0, 10) ]
     
    p = (20, 20)
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')
       
    p = (5, 5)
    if (is_inside_polygon(points = polygon1, p = p)):
      print ('Yes')
    else:
      print ('No')
 
    polygon2 = [ (0, 0), (5, 0), (5, 5), (3, 3) ]
     
    p = (3, 3)
    if (is_inside_polygon(points = polygon2, p = p)):
      print ('Yes')
    else:
      print ('No')
       
    p = (5, 1)
    if (is_inside_polygon(points = polygon2, p = p)):
      print ('Yes')
    else:
      print ('No')
       
    p = (8, 1)
    if (is_inside_polygon(points = polygon2, p = p)):
      print ('Yes')
    else:
      print ('No')
     
    polygon3 = [ (0, 0), (10, 0), (10, 10), (0, 10) ] 
     
    p = (-1, 10)
    if (is_inside_polygon(points = polygon3, p = p)):
      print ('Yes')
    else:
      print ('No')


#4##################################################################################################################
'''
Jarvis’s Algorithm or Wrapping

Given a set of points in the plane. the convex hull of the set is the smallest convex polygon that contains all the points of it.

sol.
start from the left most point and add it to the res set.
connect the last item on the res set to the next available point and see if there is any point to the left of that line
if yes then disregard that point and repeat prev state until you find one
if you found one add it to the res set.
if you find a point which on the same line add that too.

to do all this:
he idea of Jarvis’s Algorithm is simple, we start from the leftmost point (or point with minimum x coordinate value) 
and we keep wrapping points in counterclockwise direction. The big question is, given a point p as current point, how 
to find the next point in output? The idea is to use orientation() here. Next point is selected as the point that beats 
all other points at counterclockwise orientation, i.e., next point is q if for any other point r, we have 
“orientation(p, q, r) = counterclockwise”. 

https://www.youtube.com/watch?v=Vu84lmMzP2o
'''

# point class with x, y as point  
class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
  
def Left_index(points): 
      
    ''' 
    Finding the left most point 
    '''
    minn = 0
    for i in range(1,len(points)): 
        if points[i].x < points[minn].x: 
            minn = i 
        elif points[i].x == points[minn].x: 
            if points[i].y > points[minn].y: 
                minn = i 
    return minn 
  
def orientation(p, q, r): 
    ''' 
    To find orientation of ordered triplet (p, q, r).  
    The function returns following values  
    0 --> p, q and r are colinear  
    1 --> Clockwise  
    2 --> Counterclockwise  
    '''
    val = (q.y - p.y) * (r.x - q.x) - \ 
          (q.x - p.x) * (r.y - q.y) 
  
    if val == 0: 
        return 0
    elif val > 0: 
        return 1
    else: 
        return 2
  
def convexHull(points, n): 
      
    # There must be at least 3 points  
    if n < 3: 
        return
  
    # Find the leftmost point 
    l = Left_index(points) 
  
    hull = [] 
      
    ''' 
    Start from leftmost point, keep moving counterclockwise  
    until reach the start point again. This loop runs O(h)  
    times where h is number of points in result or output.  
    '''
    p = l 
    q = 0
    while(True): 
          
        # Add current point to result  
        hull.append(p) 
  
        ''' 
        Search for a point 'q' such that orientation(p, x,  
        q) is counterclockwise for all points 'x'. The idea  
        is to keep track of last visited most counterclock-  
        wise point in q. If any point 'i' is more counterclock-  
        wise than q, then update q.  
        '''
        q = (p + 1) % n 
  
        for i in range(n): 
              
            # If i is more counterclockwise  
            # than current q, then update q  
            if(orientation(points[p],  
                           points[i], points[q]) == 2): 
                q = i 
  
        ''' 
        Now q is the most counterclockwise with respect to p  
        Set p as q for next iteration, so that q is added to  
        result 'hull'  
        '''
        p = q 
  
        # While we don't come to first point 
        if(p == l): 
            break
  
    # Print Result  
    for each in hull: 
        print(points[each].x, points[each].y) 
  
# Driver Code 
points = [] 
points.append(Point(0, 3)) 
points.append(Point(2, 2)) 
points.append(Point(1, 1)) 
points.append(Point(2, 1)) 
points.append(Point(3, 0)) 
points.append(Point(0, 0)) 
points.append(Point(3, 3)) 
  
convexHull(points, len(points)) 
