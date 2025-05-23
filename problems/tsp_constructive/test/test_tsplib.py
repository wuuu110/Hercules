import numpy as np
from scipy.spatial import distance_matrix
from copy import copy
from tqdm import tqdm



names = ["ts225", "rat99", "rl1889", "u1817", "d1655", "bier127", "lin318", "eil51", "d493", "kroB100", "kroC100", "ch130", "d657", "kroA150", "u724", "pr264", "pr226", "pr439"]

opt = {
    'ts225': 126643,
    'rat99': 1211,
    'rl1889': 316536,
    'u1817': 57201,
    'd1655': 62128,
    'bier127': 118282,
    'lin318': 42029,
    'eil51': 426,
    'd493': 35002,
    'kroB100': 22141,
    'kroC100': 20749,
    'ch130': 6110,
    'd657': 48912,
    'kroA150': 26524,
    'u724': 41910,
    'pr264': 49135,
    'pr226': 80369,
    'pr439': 107217
 }



#please define the heuristic
def select_next_node()



def eval_heuristic(node_positions: np.ndarray, start_node: int) -> float:
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in tqdm(range(problem_size - 1)):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj


if __name__ == '__main__':
    for name in names:
        with open(f'tsplib/{name}.tsp') as f:
            lines = f.readlines()

        # Parse the data
        data = lines[6:-1]
        data = [x.strip().split() for x in data]
        data = [[float(x) for x in row[1:]] for row in data]

        # Scale the data to [0, 1]^2 to align with the training data
        data = np.array(data)
        scale = max(np.max(data, axis=0) - np.min(data, axis=0))
        data = (data - np.min(data, axis=0)) / scale

        # Evaluate the heuristic
        objs = []
        for start_node in range(3):
            obj = eval_heuristic(data, start_node) * scale
            objs.append(obj)
        mean, std = np.mean(objs), np.std(objs)
        print(name)
        print(f"\tObjective: {mean}+-{std}")
        print(f"\tOpt. Gap: {(mean - opt[name]) / opt[name] * 100}%")
        print()
