from aco import ACO
import numpy as np
import torch
import logging

import numpy as np

N_ANTS = 10
N_ITERATIONS = [1, 10, 20, 30, 40, 50]


def solve(prize: np.ndarray, weight: np.ndarray,heuristics):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)

    results = []
    for i in range(len(N_ITERATIONS)):
        if i == 0:
            obj, _ = aco.run(N_ITERATIONS[i])
        else:
            obj, _ = aco.run(N_ITERATIONS[i] - N_ITERATIONS[i - 1])
        results.append(obj.item())
    return results

def mkp_aco(heuristics):
    for problem_size in [120,500,1000]:
        print(problem_size)
        dataset_path = f"dataset/test{problem_size}_dataset.npz"
        dataset = np.load(dataset_path)
        prizes, weights = dataset['prizes'], dataset['weights']
        n_instances = prizes.shape[0]
        logging.info(f"[*] Evaluating {dataset_path}")

        objs = []
        for i, (prize, weight) in enumerate(zip(prizes, weights)):
            obj = solve(prize, weight, heuristics)
            objs.append(obj)

        # Average objective value for all instances
        mean_obj = np.mean(objs, axis=0)
        for i, obj in enumerate(mean_obj):
            print(f"[*] Average for {problem_size}, {N_ITERATIONS[i]} iterations: {obj}")


def main():
     # please define the heuristic function
    mkp_aco(heuristic)


if __name__ == "__main__":
        main()
