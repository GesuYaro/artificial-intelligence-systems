import pandas
import heapq


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

df2 = pandas.read_csv('../roads_with_euristic.txt', sep=' ')
dist_to_endcity = [0] * len(roads)
for i, row in df2.iterrows():
    dist = row['Расстояние,км']
    city_ind = cities[row['Город']]
    dist_to_endcity[city_ind] = dist

##########################################################


def bfs(start):
    global distances
    global roads
    distances[start] = 0
    queue = [[0, start]]
    while queue:
        v = heapq.heappop(queue)[1]
        for i in range(len(roads[v])):
            if roads[v][i] != 0:
                if distances[i] == -1:
                    distances[i] = distances[v] + roads[v][i]
                    heapq.heappush(queue, [dist_to_endcity[i], i])


bfs(city1)
print(distances[city2])
