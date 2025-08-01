import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import Counter

T = 0
SPAWN_RATES = {"low": (5, 15), "high": (25, 50)}
CAR_COUNTER = 0
random.seed(24)

G = nx.DiGraph()

# edges = [
#     ("A", "B", {"base_time": 10, "capacity": 0, "jam_factor": 1, "active_cars": {}}),
#     (
#         "A",
#         "C",
#         {"base_time": 30, "capacity": 1000000, "jam_factor": 0, "active_cars": {}},
#     ),
#     (
#         "B",
#         "D",
#         {"base_time": 30, "capacity": 1000000, "jam_factor": 0, "active_cars": {}},
#     ),
#     ("C", "D", {"base_time": 10, "capacity": 0, "jam_factor": 1, "active_cars": {}}),
#     (
#         "B",
#         "C",
#         {"base_time": 1, "capacity": 1000000, "jam_factor": 0, "active_cars": {}},
#     ),
# ]
edges = [
    (
        "A",
        "B",
        {
            "base_time": 0,
            "capacity": 0,
            "jam_factor": 0.01,
            "active_cars": {},
        },
    ),
    (
        "A",
        "C",
        {
            "base_time": 4,
            "capacity": 1000000,
            "jam_factor": 0,
            "active_cars": {},
        },
    ),
    (
        "B",
        "D",
        {
            "base_time": 4,
            "capacity": 1000000,
            "jam_factor": 0,
            "active_cars": {},
        },
    ),
    (
        "C",
        "D",
        {
            "base_time": 0,
            "capacity": 0,
            "jam_factor": 0.01,
            "active_cars": {},
        },
    ),
    # (
    #     "B",
    #     "C",
    #     {
    #         "base_time": 0.001,
    #         "capacity": 100000,
    #         "jam_factor": 0,
    #         "active_cars": {},
    #     },
    # ),
]
G.add_edges_from(edges)

cars = {}

edge_activity = {(u, v): [] for u, v in G.edges()}


def get_travel_time(attrs_dict):
    base_time = attrs_dict["base_time"]
    capacity = attrs_dict["capacity"]
    active_cars_count = len(attrs_dict["active_cars"])
    jam_factor = attrs_dict["jam_factor"]
    return base_time + max(active_cars_count - capacity, 0) * jam_factor


def update_weights():
    for u, v, attrs in G.edges(data=True):
        G[u][v]["weight"] = get_travel_time(attrs)


def spawn_cars(current_time, total_cars):
    global CAR_COUNTER
    for _ in range(total_cars):
        CAR_COUNTER += 1
        car_id = CAR_COUNTER
        car = {
            "id": car_id,
            "start": "A",
            "end": "D",
            "path": tuple(nx.dijkstra_path(G, "A", "D")),
            "path_index": 0,
            "current_edge": (),
            "position": 0.0,
            "enter_time": current_time,
            "total_time": 0,
            "status": "en-route",
        }
        cars[car_id] = car

        path = car["path"]
        u, v = path[0], path[1]
        G[u][v]["active_cars"][car_id] = 0.0


def update_cars(t):
    arrived = []
    for car_id, car in cars.items():
        if car["status"] != "en-route":
            continue
        idx = car["path_index"]
        u, v = car["path"][idx], car["path"][idx + 1]
        edge = G[u][v]
        weight = edge["weight"]
        progress = 1 / weight if weight != 0 else 0.0

        current_pos = car["position"]
        new_pos = current_pos + progress

        if new_pos >= 1.0:
            del edge["active_cars"][car_id]

            current_node = car["path"][idx + 1]
            if current_node == car["end"]:
                # trip complete
                car["status"] = "arrived"
                car["total_time"] = t - car["enter_time"]
                arrived.append(car_id)
            else:
                try:
                    new_path = nx.dijkstra_path(G, current_node, car["end"])
                except nx.NetworkXNoPath:
                    car["status"] = "stuck"
                    continue
                car["path"] = car["path"][: idx + 1] + tuple(new_path)
                car["path_index"] += 1
                car["position"] = 0.0
                next_u, next_v = (
                    car["path"][car["path_index"]],
                    car["path"][car["path_index"] + 1],
                )
                G[next_u][next_v]["active_cars"][car_id] = 0.0
        else:
            car["position"] = new_pos
            edge["active_cars"][car_id] = new_pos

    return arrived


while True:
    update_weights()

    # log active cars per edge for plotting
    for u, v in G.edges():
        edge_activity[(u, v)].append(len(G[u][v]["active_cars"]))

    if T == 0:
        spawn_cars(T, 400)

    # if T <= 20:
    #     rate = random.randint(*SPAWN_RATES["low"])
    #     spawn_cars(T, rate)
    # elif T >= 21 and T <= 80:
    #     rate = random.randint(*SPAWN_RATES["high"])
    #     spawn_cars(T, rate)

    arrived = update_cars(T)

    if T % 1 == 0:
        print(f"\nTime: {T}")
        print(f"Total cars: {len(cars)}")
        print(f"Arrived: {len([c for c in cars.values() if c['status'] == 'arrived'])}")
        print(f"Current edge loads:")
        for u, v, data in G.edges(data=True):
            print(
                f"  {u}->{v} | active cars: {len(data['active_cars'])} | weight: {round(data['weight'], 3)}"
            )

        if all(car["status"] == "arrived" for car in cars.values()):
            print("Simulation complete.")
            print("Summary Statistics:")
            avg_travel_time = sum(car["total_time"] for car in cars.values()) / len(
                cars
            )
            print("Average travel time:", avg_travel_time)
            all_paths = [car["path"] for car in cars.values()]
            path_counter = Counter(all_paths)
            print("\nPaths taken (by frequency):")
            for path, freq in path_counter.most_common():
                print(f"Path: {' -> '.join(path)} | Count: {freq}")

            print("\nAverage travel time per path:")
            path_times = {}
            for car in cars.values():
                if car["status"] == "arrived":
                    path_times.setdefault(car["path"], []).append(car["total_time"])
            for path, times in sorted(
                path_times.items(), key=lambda x: len(x[1]), reverse=True
            ):
                avg_time = sum(times) / len(times)
                print(
                    f"Path: {' -> '.join(path)} | Avg travel time: {avg_time:.2f} | Trips: {len(times)}"
                )
            break

    T += 1

# plt.figure(figsize=(10, 6))
# for (u, v), counts in edge_activity.items():
#     plt.plot(range(len(counts)), counts, label=f"{u}->{v}")

# plt.xlabel("Time Step")
# plt.ylabel("Number of Active Cars")
# plt.title("Active Cars on Each Edge Over Time")
# plt.legend()
# plt.tight_layout()
# plt.show()
