import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx


class TrafficEnv(gym.Env):
    def __init__(
        self, max_cars=1, graph_edges=None, car_origins=None, car_destinations=None
    ):
        super().__init__()
        self.G = nx.DiGraph()
        self.edges = (
            graph_edges
            if graph_edges is not None
            else [
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
                (
                    "B",
                    "C",
                    {
                        "base_time": 0.1,
                        "capacity": 100000,
                        "jam_factor": 0,
                        "active_cars": {},
                    },
                ),
            ]
        )
        self.edge_list = [(u, v) for u, v, _ in self.edges]
        self.G.add_edges_from(self.edges)
        self.max_cars = max_cars
        self.cars = {}
        self.car_counter = 0
        self.T = 0

        self.car_origins = (
            car_origins if car_origins is not None else ["A"] * self.max_cars
        )
        self.car_destinations = (
            car_destinations if car_destinations is not None else ["D"] * self.max_cars
        )

        max_choices_per_node = max(
            [len(list(self.G.successors(node))) for node in self.G.nodes()] + [1]
        )
        self.action_space = spaces.MultiDiscrete([max_choices_per_node] * self.max_cars)
        self.observation_space = spaces.Box(
            low=0,
            high=1e6,
            shape=(len(self.edge_list) * 2 + self.max_cars * 3,),
            dtype=np.float32,
        )
        self.reset()

    def get_travel_time(self, attrs_dict):
        base_time = attrs_dict["base_time"]
        capacity = attrs_dict["capacity"]
        active_cars_count = len(attrs_dict["active_cars"])
        jam_factor = attrs_dict["jam_factor"]
        return base_time + max(active_cars_count - capacity, 0) * jam_factor

    def update_weights(self):
        for u, v, attrs in self.G.edges(data=True):
            self.G[u][v]["weight"] = self.get_travel_time(attrs)

    def reset(self, seed=None, options=None):
        self.G.clear()
        self.G.add_edges_from(self.edges)
        for _, _, d in self.G.edges(data=True):
            d["active_cars"] = {}
        self.cars = {}
        self.car_counter = 0
        self.T = 0

        self.update_weights()

        for i in range(self.max_cars):
            car_id = i + 1
            start = self.car_origins[i] if i < len(self.car_origins) else "A"
            end = self.car_destinations[i] if i < len(self.car_destinations) else "D"
            self.update_weights()
            try:
                init_path = tuple(nx.dijkstra_path(self.G, start, end))
            except nx.NetworkXNoPath:
                init_path = (start, end)
            car = {
                "id": car_id,
                "start": start,
                "end": end,
                "path": init_path,
                "path_index": 0,
                "current_edge": (
                    (init_path[0], init_path[1]) if len(init_path) > 1 else ()
                ),
                "position": 0.0,
                "enter_time": self.T,
                "total_time": 0,
                "status": "en-route",
            }
            self.cars[car_id] = car
            if car["current_edge"]:
                u, v = car["current_edge"]
                self.G[u][v]["active_cars"][car_id] = 0.0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        edge_features = []
        for u, v in self.edge_list:
            edge = self.G[u][v]
            edge_features.extend([edge["weight"], len(edge["active_cars"])])
        car_features = []
        node_to_int = {node: i for i, node in enumerate(self.G.nodes())}
        status_to_int = {"en-route": 1, "arrived": 0, "stuck": -1}
        for car_id in range(1, self.max_cars + 1):
            car = self.cars.get(car_id)
            if car:
                car_features.extend(
                    [
                        node_to_int.get(car["path"][car["path_index"]], 0),
                        car["position"],
                        status_to_int.get(car["status"], -1),
                    ]
                )
            else:
                car_features.extend([0, 0.0, -1])
        obs = np.array(edge_features + car_features, dtype=np.float32)
        return obs

    def step(self, actions):
        system_travel_time_before = sum(
            [
                (
                    car["total_time"]
                    if car["status"] == "arrived"
                    else (self.T - car["enter_time"])
                )
                for car in self.cars.values()
            ]
        )

        self.update_weights()

        arrived = []
        for car_id, car in self.cars.items():
            if car["status"] != "en-route":
                continue
            idx = car["path_index"]
            path = car["path"]
            if idx + 1 >= len(path):
                continue
            u, v = path[idx], path[idx + 1]
            edge = self.G[u][v]
            weight = edge["weight"]
            progress = 1 / weight if weight != 0 else 0.0
            current_pos = car["position"]
            new_pos = current_pos + progress

            if new_pos >= 1.0:
                del edge["active_cars"][car_id]
                car["path_index"] += 1
                car["position"] = 0.0
                if car["path"][car["path_index"]] == car["end"]:
                    car["status"] = "arrived"
                    car["total_time"] = self.T - car["enter_time"]
                    arrived.append(car_id)
                    car["current_edge"] = ()
                else:
                    current_node = car["path"][car["path_index"]]
                    choices = list(self.G.successors(current_node))
                    if choices:
                        action = (
                            actions[car_id - 1] % len(choices)
                            if len(choices) > 1
                            else 0
                        )
                        next_node = choices[action]

                        try:
                            remaining_path = nx.dijkstra_path(
                                self.G, next_node, car["end"]
                            )
                            current_node_in_path = car["path"][car["path_index"]]
                            car["path"] = car["path"][: car["path_index"] + 1] + tuple(
                                remaining_path
                            )
                            assert (
                                car["path"][car["path_index"]] == current_node_in_path
                            )
                        except nx.NetworkXNoPath:
                            if self.G.has_edge(next_node, car["end"]):
                                car["path"] = car["path"][: car["path_index"] + 1] + (
                                    next_node,
                                    car["end"],
                                )
                            else:
                                car["status"] = "stuck"
                                car["current_edge"] = ()
                                continue

                        car["current_edge"] = (current_node, next_node)
                        self.G[current_node][next_node]["active_cars"][car_id] = 0.0
                    else:
                        car["status"] = "stuck"
                        car["current_edge"] = ()
            else:
                car["position"] = new_pos
                edge["active_cars"][car_id] = new_pos

        self.T += 1

        system_travel_time_after = sum(
            [
                (
                    car["total_time"]
                    if car["status"] == "arrived"
                    else (self.T - car["enter_time"])
                )
                for car in self.cars.values()
            ]
        )

        obs = self._get_obs()
        reward = -(system_travel_time_after - system_travel_time_before)
        terminated = all(car["status"] == "arrived" for car in self.cars.values())
        truncated = False
        info = {"arrived": arrived}

        if self.T > 200:
            truncated = True

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        print(f"Time: {self.T}")
        for u, v, data in self.G.edges(data=True):
            print(
                f"  {u}->{v} | active cars: {len(data['active_cars'])} | weight: {round(data['weight'], 3)}"
            )

    def get_final_paths(self):
        paths = {}
        for car_id, car in self.cars.items():
            if car["status"] == "arrived":
                paths[car_id] = {"path": car["path"], "total_time": car["total_time"]}
        return paths
