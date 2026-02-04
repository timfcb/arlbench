from scipy.spatial.distance import cityblock
from statistics import mean
from itertools import product

n = 10 


# Erzeuge all pairs 

all_positions = [[i,j] for i in range(1, n+1) for j in range(1, n+1)]
print(all_positions)

print('ALl pairs')
all_pairs = list(product(all_positions, all_positions))

print(all_pairs)


print('All pairs converted')

converted = [list(t) for t in all_pairs]

print(converted)
all_distances = [cityblock(pair[0], pair[1]) for pair in all_pairs]

print(all_distances)

print(len(all_distances))

remove_zeros = [x for x in all_distances if x != 0]

print(len(remove_zeros))

print(sum(remove_zeros))

print(f'AVG is: {sum(remove_zeros)/len(remove_zeros)}')

#avg_distances = mean(all_distances)

#print(avg_distances)