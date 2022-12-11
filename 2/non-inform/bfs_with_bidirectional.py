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

##########################################################
visited = set()
back_visited = set()


def bidirectional_bfs(start, finish):
    global roads
    distances = [-1] * len(cities)
    back_distances = [-1] * len(cities)
    distances[start] = 0
    back_distances[start] = 0
    queue = [start]
    back_queue = [finish]
    while queue and back_queue:
        v = queue.pop(0)
        visited.add(v)
        back_v = back_queue.pop(0)
        back_visited.add(back_v)
        if v in back_visited and back_v in visited:
            return distances[v] + back_distances[v]
        for i in range(len(roads[v])):
            if roads[v][i] != 0:
                if distances[i] == -1:
                    queue.append(i)
                    distances[i] = distances[v] + roads[v][i]
        for i in range(len(roads[back_v])):
            if roads[back_v][i] != 0:
                if back_distances[i] == -1:
                    back_queue.append(i)
                    back_distances[i] = back_distances[back_v] + roads[back_v][i]


ans = bidirectional_bfs(city1, city2)
print(ans)

