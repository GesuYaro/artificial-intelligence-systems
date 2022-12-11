import pandas


def print_roads(roads):
    for r in roads:
        print(r)


df = pandas.read_csv('../roads.txt', sep=' ')
all_cities = set(df['Город1'].tolist() + df['Город2'].tolist())
all_cities = list(zip(all_cities, range(len(all_cities))))
cities = {x: y for x, y in all_cities}
cities_rev = {y: x for x, y in all_cities}

city1 = cities['Самара']
city2 = cities['Ярославль']
distances = [-1] * len(cities)

roads = [0] * len(cities)
for i in range(len(roads)):
    roads[i] = [0] * len(cities)

for i, row in df.iterrows():
    dist = row['Расстояние,км']
    city1_ind = cities[row['Город1']]
    city2_ind = cities[row['Город2']]
    roads[city1_ind][city2_ind] = dist
    roads[city2_ind][city1_ind] = dist

print_roads(roads)
print(city1, city2)

##########################################

max_depth = 5


def dfs(curr, prev, finish, curr_depth, is_start=False):
    global roads
    global distances
    global max_depth
    if curr_depth > max_depth:
        return False
    if is_start:
        distances[curr] = 0
    else:
        distances[curr] = distances[prev] + roads[curr][prev]
    if curr == finish:
        return True
    for i in range(len(roads[curr])):
        if roads[curr][i] != 0:
            if distances[i] == -1:
                if dfs(i, curr, finish, curr_depth + 1):
                    return True
    return False


if dfs(city1, city1, city2, 0, True):
    print(distances[city2])
else:
    print("Not reached")
