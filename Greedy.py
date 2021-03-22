#1##########################################################################################

'''
Job sequencing O(n^2)

greedy algorithm

we get a list of jobs with IDs, deadline (in unit of time), and profit.

JobID  Deadline  Profit
  a      4        20   
  b      1        10
  c      1        40  
  d      1        30
  
we want to maximize profit if only one job can be scheduled at a time.

solution is to sort the table based on profit. 
take the highest value and check the deadline. if it's 3 for instance, you have to put the ID
in an array on the third slot. if third slot is full, go one back until you cannot go further back.
if there is no place to put it then we should skip the job  
'''

def jobSequencing(jobs):
    n = len(jobs)
    temp_sequence = [0 for i in range(n)]
    jobs_sorted = sortJobsOnProfit(jobs)

    for i in range(n-1, -1, -1):

        j = jobs_sorted[i][1] - 1
      
        while j >= 0 :
            if temp_sequence[j] == 0:
                temp_sequence[j] = jobs_sorted[i][0]
                break
            j -= 1
    sequence = []
    for e in temp_sequence :
        if e != 0:
            sequence.append(e)
    return sequence
    
def sortJobsOnProfit(jobs):
    return sorted(jobs, key = lambda j : j[2])
    
    
jobs = [['a',4,20], 
        ['b', 1, 10],
        ['c', 1, 40],
        ['d', 1, 30]
        ]    

print("Job Sequence to Maximize Profit : ", jobSequencing(jobs))