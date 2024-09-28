from aco import ACO
import numpy as np
import logging
from gen_inst import BPPInstance, load_dataset, dataset_conf

N_ANTS = 30
N_ITERATIONS = [1, 10, 30, 50, 80, 100]

def solve(inst: BPPInstance, heuristics):
    heu = heuristics(inst.demands.copy(), inst.capacity)  # normalized in ACO
    assert tuple(heu.shape) == (inst.n, inst.n)
    assert 0 < heu.max() < np.inf
    aco = ACO(inst.demands, heu.astype(float), capacity=inst.capacity, n_ants=N_ANTS, greedy=False)

    results = []
    for i in range(len(N_ITERATIONS)):
        if i == 0:
            obj, _ = aco.run(N_ITERATIONS[i])
        else:
            obj, _ = aco.run(N_ITERATIONS[i] - N_ITERATIONS[i - 1])
        results.append(obj)

    return results
def aco_bpp(heuristic):
    for problem_size in [120, 500, 1000]:
        dataset_path = f"dataset/val{problem_size}_dataset.npz"
        dataset = load_dataset(dataset_path)
        n_instances = dataset[0].n
        logging.info(f"[*] Evaluating {dataset_path}")

        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance, heuristics_human)  # expert-defined heuristics
            objs.append(obj)

        # Average objective value for all instances
        mean_obj = np.mean(objs, axis=0)
        for i, obj in enumerate(mean_obj):
            print(f"[*] Average for {problem_size}, {N_ITERATIONS[i]} iterations: {obj}")
def main():
   # please define the heuristic function
    aco_bpp(heuristic)

if __name__ == "__main__":
        main()
