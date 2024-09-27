from aco import ACO
import numpy as np
import logging
from gen_inst import BPPInstance, load_dataset, dataset_conf

N_ANTS = 30
N_ITERATIONS = [1, 10, 30, 50, 80, 100]


def pppp1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))
    max_combined_size = capacity

    for i in range(n):
        for j in range(i + 1, n):
            combined_size = demand[i] + demand[j]
            if combined_size <= capacity:
                # Diversity factor encourages pairing items of different sizes
                diversity_factor = 1 - (abs(demand[i] - demand[j]) / max_combined_size)
                # Waste penalty discourages leaving too much empty space in a bin
                waste = capacity - combined_size
                waste_penalty = waste / capacity
                # Full bin utilization factor promotes filling the bin to its capacity
                utilization_factor = combined_size / max_combined_size

                # Penalize waste more heavily for larger items
                size_factor = min(demand[i], demand[j]) / max_combined_size
                adjusted_waste_penalty = waste_penalty * (1 + size_factor)

                # Encourage full bins more for diverse pairs
                adjusted_diversity_factor = diversity_factor * (1 + utilization_factor)

                # Combine these factors into a single score
                score = adjusted_diversity_factor / (adjusted_waste_penalty + 1e-6)

                # Sparsify the matrix by setting lower scores to zero
                if score < 0.5:
                    score = 0

                heuristics_matrix[i, j] = heuristics_matrix[j, i] = score

    return heuristics_matrix


def pppp2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = demand[i] + demand[j]
            if pair_sum <= capacity:
                # Exact fit should not be penalized, hence a special case
                if pair_sum == capacity:
                    score = 1.0
                else:
                    # Calculate the score based on the complementarity
                    # Avoid negative scores which would happen when the pair is far from filling the bin
                    complementarity = (capacity - pair_sum) / capacity
                    # Penalize the score based on how much space is left unused
                    penalty = 1 - complementarity
                    score = (pair_sum / capacity) * (1 - penalty)
            else:
                # If the sum of the pair exceeds the capacity, set a low score
                score = 0

            heuristics_matrix[i, j] = heuristics_matrix[j, i] = score

    # Sparsify the matrix by setting less promising edges (values below a threshold) to zero
    threshold = 0.7  # This threshold can be tuned
    heuristics_matrix[heuristics_matrix < threshold] = 0

    return heuristics_matrix


def pppp3(demand: np.ndarray, capacity: int) -> np.ndarray:
    normalized_demand = demand / capacity
    heuristics_matrix = np.outer(normalized_demand, normalized_demand)
    combined_demands = demand[:, None] + demand

    penalty = -np.inf
    heuristics_matrix[combined_demands > capacity] = penalty

    complementarity = np.where(combined_demands <= capacity,
                               np.minimum.outer(normalized_demand, 1 - normalized_demand), 0)
    feasibility_factor = 1 + complementarity[combined_demands <= capacity]
    heuristics_matrix[combined_demands <= capacity] *= feasibility_factor

    np.fill_diagonal(heuristics_matrix, 0)

    # Sparsify the matrix more aggressively by setting the lower 50% of scores to zero
    threshold = np.percentile(heuristics_matrix[heuristics_matrix != penalty], 50)
    heuristics_matrix[heuristics_matrix < threshold] = 0

    # Enhance normalization to ensure all values are between 0 and 1
    max_value = np.max(heuristics_matrix[heuristics_matrix > 0])
    min_value = np.min(heuristics_matrix[heuristics_matrix > 0])
    heuristics_matrix[heuristics_matrix > 0] = (heuristics_matrix[heuristics_matrix > 0] - min_value) / (
                max_value - min_value)

    return heuristics_matrix


def hercules1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristic_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            total_demand = demand[i] + demand[j]
            if total_demand <= capacity:
                # Quadratic utilization rewards
                utilization = total_demand / capacity
                score = utilization ** 2

                # Waste penalty
                empty_space = capacity - total_demand
                penalty = 1 / ((empty_space + 1) ** 2) if empty_space > 0 else 1

                # Diversity bonus
                size_difference = abs(demand[i] - demand[j])
                diversity_bonus = 1 - (size_difference / capacity) ** 2

                # Combine score, penalty, and bonus
                combined_score = score * penalty * diversity_bonus

                # Assign the combined score
                heuristic_matrix[i, j] = heuristic_matrix[j, i] = combined_score

    # Normalize the heuristic matrix
    max_value = np.max(heuristic_matrix)
    if max_value > 0:
        heuristic_matrix /= max_value

    # Sparsify the matrix
    heuristic_matrix[heuristic_matrix < 0.05] = 0

    return heuristic_matrix
def hercules2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            combined_size = demand[i] + demand[j]
            if combined_size <= capacity:
                # Reward pairs that use the bin space most efficiently
                reward_factor = combined_size / capacity
                # Penalize wasted space disproportionately
                unused_space = capacity - combined_size
                penalty_factor = np.exp(-unused_space)
                # Highlight the most promising pairs
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = reward_factor * penalty_factor
            else:
                # Sparsify the matrix by setting unpromising elements to zero
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = 0

    # Normalize the matrix to make values more comparable
    heuristics_matrix /= heuristics_matrix.max()

    return heuristics_matrix
def hercules3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    # Calculate the combined size of every possible pair of items
    combined_sizes = demand[:, None] + demand

    # Identify pairs that can fit into a single bin
    can_fit_together = combined_sizes <= capacity

    # Calculate remaining space if the items are placed together
    remaining_space = capacity - combined_sizes
    remaining_space[~can_fit_together] = 0  # Set remaining space to 0 for pairs that cannot fit together

    # Use a heuristic that prioritizes pairs with smaller remaining space using inverse scaling
    scores = 1 / (1 + remaining_space) if remaining_space.any() else remaining_space

    # Enhance the score of pairs that are a tight fit (remaining space close to 0)
    tight_fit_bonus = 1 / (1 + remaining_space**2)
    scores = scores * tight_fit_bonus

    # Apply the scores only to the pairs that can fit together
    heuristics_matrix[can_fit_together] = scores[can_fit_together]

    # Normalize the scores to be between 0 and 1
    heuristics_matrix /= heuristics_matrix.max()

    # Sparsify the matrix by setting very low scores to zero
    heuristics_matrix[heuristics_matrix < 1e-6] = 0

    return heuristics_matrix


def random1(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate how much space would be left if items i and j were placed in the same bin
                space_left = capacity - demand[i] - demand[j]
                # The more negative the space_left, the less promising it is to put them together
                # We inverse it to make it more intuitive (more positive, more promising)
                # Adding a small constant to avoid division by zero
                heuristics_matrix[i, j] = 1 / (space_left + 1e-6) if space_left >= 0 else 0

    return heuristics_matrix


def random2(demand, capacity):
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))

    # Iterate over all pairs of items
    for i in range(n):
        for j in range(i + 1, n):
            # If both items can fit in the same bin
            if demand[i] + demand[j] <= capacity:
                # The more space is left in the bin after packing both items, the more promising the pair is
                space_left = capacity - (demand[i] + demand[j])
                # Promising score could be inversely proportional to the remaining space (the less space left, the higher the score)
                # To ensure non-zero scores, we add a small constant (e.g., 0.1)
                score = 1 / (space_left + 0.1)
                heuristics_matrix[i, j] = score
                heuristics_matrix[j, i] = score

    return heuristics_matrix
def random3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if demand[i] + demand[j] <= capacity:
                # Reward pairs that can fit together with a higher score
                # The score is inversely proportional to the remaining space
                # after placing both items together
                remaining_space = capacity - demand[i] - demand[j]
                heuristics_matrix[i][j] = heuristics_matrix[j][i] = 1.0 / (1.0 + remaining_space)
    return heuristics_matrix


def p1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                total_demand = demand[i] + demand[j]
                if total_demand <= capacity:
                    # Penalize pairs where one item is much smaller than the other to avoid asymmetric fits
                    penalty = abs(demand[i] - demand[j]) / capacity
                    # Calculate the score based on remaining space and penalize for asymmetric fits
                    remaining_space = capacity - total_demand
                    score = (1 / (1 + remaining_space)) * (1 - penalty)
                    # Sparsify the matrix by setting low score edges to zero
                    if score < 0.1:
                        score = 0
                    heuristics_matrix[i, j] = score

    return heuristics_matrix


def p2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    # Vectorize the computation of combined demands
    demand_combinations = np.add.outer(demand, demand)

    # Identify pairs that can fit in one bin
    valid_pairs = demand_combinations <= capacity

    # Calculate the score for valid pairs
    combined_demand = demand_combinations[valid_pairs]
    unused_space = capacity - combined_demand
    full_bin_bonus = 2 * capacity * (combined_demand == capacity)
    penalty_for_unused_space = unused_space ** 2 / (capacity ** 2)

    # The score is a balance of penalty for unused space and bonus for full bins
    compatibility_scores = (combined_demand / capacity) - penalty_for_unused_space + full_bin_bonus

    # Apply the scores to the matrix while symmetrizing it
    i_indices, j_indices = np.where(valid_pairs)
    heuristics_matrix[i_indices, j_indices] = compatibility_scores
    heuristics_matrix[j_indices, i_indices] = compatibility_scores

    # Sparsify the matrix by setting the least promising scores to zero
    heuristics_matrix[heuristics_matrix < 0.5] = 0

    return heuristics_matrix
def p3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    normalized_demand = demand / capacity
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            combined_size = normalized_demand[i] + normalized_demand[j]
            # High utilization bonus
            utilization = min(combined_size, 1)
            # Waste penalty, more aggressive for higher waste
            waste_penalty = (1 - utilization) ** 2
            # Score calculation: prioritize high utilization and low waste
            score = utilization / (waste_penalty + 1e-5)
            # Assign score to matrix element if the combined size is less than capacity
            if combined_size <= 1:
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = score

    # Apply aggressive thresholding to zero out less promising pairs
    threshold = np.percentile(heuristics_matrix[heuristics_matrix > 0], 95)
    heuristics_matrix[heuristics_matrix < threshold] = 0

    # Normalize the scores to emphasize the most promising pairs
    max_score = np.max(heuristics_matrix)
    if max_score > 0:
        heuristics_matrix = heuristics_matrix / max_score

    return heuristics_matrix


def reevo1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            combined_demand = demand[i] + demand[j]
            if combined_demand <= capacity:
                fill_level = combined_demand / capacity
                leftover_space = capacity - combined_demand

                # Improved similarity calculation that emphasizes the pair's volume compared to the bin capacity
                size_similarity = (demand[i] + demand[j]) / capacity

                # Improved complementarity calculation based on the harmonic mean
                complementarity = 2 * demand[i] * demand[j] / (demand[i] + demand[j])

                # Adjusted penalty factor with a smaller exponent to be less aggressive
                penalty = np.exp(-0.5 * leftover_space ** 2)

                # Combine the scoring factors
                score = fill_level * complementarity * penalty * size_similarity
                max_possible_score = 1 * 1 * np.exp(-0) * 1  # Maximum values for each factor

                # Normalize the score to be between 0 and 1
                normalized_score = score / max_possible_score

                # Setting a stricter threshold for sparsification
                threshold = 0.8
                heuristics_matrix[i][j] = heuristics_matrix[j][
                    i] = normalized_score if normalized_score >= threshold else 0

    return heuristics_matrix


def reevo2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if demand[i] + demand[j] == capacity:
                # Emphasize pairs that exactly fill the bin
                score = 1.0
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = score

    # Increase sparsity by setting a high threshold to zero out less promising pairs
    threshold = 0.95
    heuristics_matrix[heuristics_matrix < threshold] = 0

    # Scale up the scores of the promising pairs to make them stand out
    heuristics_matrix[heuristics_matrix > 0] *= 100

    return heuristics_matrix


def reevo3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if demand[i] + demand[j] <= capacity:
                combined_demand = demand[i] + demand[j]
                utilization_score = combined_demand / capacity

                # Stronger penalty for unused space when the space is large
                unused_space = capacity - combined_demand
                penalty = (unused_space ** 1.5) / (capacity ** 1.5)

                # Calculate the final score considering both utilization and penalty
                score = utilization_score - 0.5 * penalty

                # Normalize the score to account for the maximum possible score
                max_possible_score = 1 - ((capacity - max(demand)) ** 1.5) / (capacity ** 1.5)
                normalized_score = score / (max_possible_score + np.finfo(float).eps)

                # Assign the normalized score to the pair and its symmetric counterpart
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = normalized_score

    # Threshold to filter out less promising pairs
    threshold = np.percentile(heuristics_matrix[heuristics_matrix > 0], 98)
    heuristics_matrix[heuristics_matrix < threshold] = 0

    return heuristics_matrix

def new1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            pair_demand = demand[i] + demand[j]
            if pair_demand <= capacity:
                # Inverse unused space heuristic with a higher power to emphasize small gaps
                unused_space = capacity - pair_demand
                score = 1 / (1 + unused_space ** 3)

                # Exact fit bonus for pairs that perfectly fill the bin
                exact_fit_bonus = 2 if unused_space == 0 else 0
                score += exact_fit_bonus

                # Enhanced complementarity factor based on the smaller unused space
                smaller_unused_space = min(capacity - demand[i], capacity - demand[j])
                complementarity_factor = 1 - (smaller_unused_space / capacity)
                score *= (1 + 2 * complementarity_factor)

                # Normalization factor based on the harmonic mean of the item demands
                normalization_factor = 2 / (1/demand[i] + 1/demand[j])
                score *= (normalization_factor / capacity)

                # Assign the score to the matrix
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = score

    # Normalize the matrix so the highest score is 1
    max_score = np.max(heuristics_matrix)
    if max_score > 0:
        heuristics_matrix /= max_score

    # Sparsify the matrix by setting lower scores to zero with a more aggressive threshold
    threshold = np.percentile(heuristics_matrix[heuristics_matrix > 0], 90)
    heuristics_matrix[heuristics_matrix < threshold] = 0

    return heuristics_matrix


def new2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = demand[i] + demand[j]
            if pair_sum <= capacity:
                # Higher penalty factor for unused space to promote efficient packing
                penalty_factor = 10
                # Heuristic score based on used capacity and penalizing unused space
                heuristic_score = pair_sum / capacity - penalty_factor * (capacity - pair_sum) / capacity
                # Ensure non-negative score
                heuristic_score = max(0, heuristic_score)
                # Assign score symmetrically
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = heuristic_score

    # Apply a strict threshold to filter out non-promising pairs
    strict_threshold = 0.5
    heuristics_matrix[heuristics_matrix < strict_threshold] = 0

    return heuristics_matrix
def new3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            pair_capacity_usage = demand[i] + demand[j]
            if pair_capacity_usage <= capacity:
                used_capacity_ratio = pair_capacity_usage / capacity
                waste = capacity - pair_capacity_usage
                waste_penalty = np.exp(-waste) if waste > 0 else 1
                normalized_waste_penalty = waste_penalty / (1 + waste if waste > 0 else 1)
                density_reward = pair_capacity_usage / min(demand[i], demand[j])
                complementary_bonus = (capacity - pair_capacity_usage) / capacity

                # Normalizing the score with respect to the size of the items
                normalization_factor = (1 + complementary_bonus) * (1 + density_reward)
                score = (used_capacity_ratio * normalized_waste_penalty) / normalization_factor

                heuristics_matrix[i][j] = heuristics_matrix[j][i] = score

    # Penalize waste more heavily and sparsify the matrix
    waste_threshold = np.exp(-0.5 * capacity)
    avg_score = np.mean(heuristics_matrix[heuristics_matrix > 0])
    heuristics_matrix[heuristics_matrix < waste_threshold * avg_score] = 0

    return heuristics_matrix


def heuristics_human(demand: np.ndarray, capacity: int) -> np.ndarray:
    return np.tile(demand / demand.max(), (demand.shape[0], 1))


def newp1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    epsilon = 1e-6
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            combined_demand = demand[i] + demand[j]
            if combined_demand <= capacity:
                # Reward high utilization and small gaps
                utilization_factor = combined_demand / capacity
                gap_penalty = 1 if combined_demand == capacity else 1 / (capacity - combined_demand + epsilon)
                score = utilization_factor * gap_penalty

                # Reward optimal fits (items exactly filling the bin)
                if combined_demand == capacity:
                    score *= (1 + epsilon)

                # Penalize waste
                space_left = capacity - combined_demand
                waste_penalty = space_left / capacity
                score -= waste_penalty

                heuristics_matrix[i][j] = heuristics_matrix[j][i] = score
            else:
                # Penalize pairs that don't fit
                heuristics_matrix[i][j] = heuristics_matrix[j][i] = -epsilon

    return heuristics_matrix


def newp2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            combined_demand = demand[i] + demand[j]
            if combined_demand <= capacity:
                # Promote exact fits with a significantly higher reward
                exact_fit_reward = 2 if combined_demand == capacity else 1

                # Enhanced non-linear scoring for utilization, favoring fuller bins more
                utilization_score = (combined_demand / capacity) ** 2

                # Improved item complementarity - higher score for less remaining space
                space_complementarity = 1 - ((capacity - combined_demand) / capacity) ** 2

                # Combine the factors with a different weighing scheme
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = (
                        utilization_score * exact_fit_reward * space_complementarity
                )
            else:
                # Penalize for not fitting more harshly
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = -np.inf

    # Normalize scores only considering feasible pairs
    max_score = np.max(heuristics_matrix[np.isfinite(heuristics_matrix)])
    if max_score > 0:
        heuristics_matrix[np.isfinite(heuristics_matrix)] /= max_score

    # Sparsify using a more aggressive percentile threshold for only the most promising pairs
    non_zero_values = heuristics_matrix[heuristics_matrix > 0]
    if non_zero_values.size > 0:
        threshold = np.percentile(non_zero_values, 98)
        heuristics_matrix[heuristics_matrix < threshold] = 0

    return heuristics_matrix


def newp3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = len(demand)
    heuristics_matrix = np.zeros((n, n))
    sorted_indices = np.argsort(demand)
    sorted_demand = demand[sorted_indices]
    inv_demand = 1 / (sorted_demand + 1e-6)  # Small constant to avoid division by zero

    # Enhanced penalty factor for space waste
    space_waste_penalty_factor = 3.0

    for i in range(n):
        for j in range(i + 1, n):
            pair_sum = sorted_demand[i] + sorted_demand[j]
            if pair_sum <= capacity:
                remaining_space = capacity - pair_sum
                diversity_score = inv_demand[i] + inv_demand[j]
                score = diversity_score * (1 + (abs(i - j) / n))  # Reward diversity in indices

                # Enhanced bonus for exact fit
                if pair_sum == capacity:
                    score += 5
                else:
                    # Enhanced penalty for space waste
                    score /= (1 + (remaining_space ** space_waste_penalty_factor))

                # Incentivize size complementarity
                size_difference = abs(sorted_demand[i] - sorted_demand[j])
                complementarity_bonus = 1 + (size_difference / capacity)
                score *= complementarity_bonus

                # Additional penalty for similar-sized items to promote diversity
                if size_difference < capacity * 0.1:
                    score *= 0.9

                heuristics_matrix[sorted_indices[i], sorted_indices[j]] = score
                heuristics_matrix[sorted_indices[j], sorted_indices[i]] = score

    return heuristics_matrix
def eoh1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    sorted_indices = np.argsort(demand)[::-1]

    for i in range(n):
        for j in range(i+1, n):
            idx_i, idx_j = sorted_indices[i], sorted_indices[j]
            if demand[idx_i] + demand[idx_j] <= capacity:
                # Adjust the score to prioritize pairs with less remaining space
                remaining_space = capacity - (demand[idx_i] + demand[idx_j])
                # Change the scoring function to have a steeper curve to penalize larger remaining spaces more
                heuristics_matrix[idx_i][idx_j] = heuristics_matrix[idx_j][idx_i] = 1 / (1 + remaining_space**2)
    return heuristics_matrix


def eoh2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            if demand[i] + demand[j] <= capacity:
                # Modified scoring: a decreasing exponential function of the remaining capacity
                remaining_capacity = capacity - (demand[i] + demand[j])
                score = np.exp(-remaining_capacity)  # The smaller the remaining capacity, the higher the score
                heuristics_matrix[i, j] = heuristics_matrix[j, i] = score

    return heuristics_matrix
def eoh3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    heuristics_matrix = np.zeros((n, n))

    # First fit a single item into a bin and then see the remaining space
    # Encourage items that can fill the bin more completely when combined
    for i in range(n):
        for j in range(i + 1, n):
            single_i_space_left = capacity - demand[i]
            if demand[j] <= single_i_space_left:
                combined_space_left = single_i_space_left - demand[j]
                # Use inverse of remaining space to indicate desirability
                # (the less space left, the higher the "score")
                score = 1 / (combined_space_left + 1e-6)  # Adding a small constant to avoid division by zero
                heuristics_matrix[i, j] = score
                heuristics_matrix[j, i] = score

    return heuristics_matrix
def llamaeoh1(demand: np.ndarray, capacity: int) -> np.ndarray:
    demand_ratio = demand / capacity
    return np.tile(np.power(demand_ratio, 2), (demand.shape[0], 1)) * (1 - demand_ratio[:, np.newaxis])
def llamaeoh2(demand: np.ndarray, capacity: int) -> np.ndarray:
    demand_ratio = demand / capacity
    return np.tile(demand_ratio, (demand.shape[0], 1)) * (1 - demand_ratio[:, np.newaxis])
def llamaeoh3(demand: np.ndarray, capacity: int) -> np.ndarray:
    demand_ratio = demand / capacity
    return np.tile(demand_ratio, (demand.shape[0], 1)) * (1 - demand_ratio[:, np.newaxis])
def llamareevo1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    demand_ratio = np.tile(demand / capacity, (n, 1))
    symmetric_ratio = demand_ratio + demand_ratio.T
    normalized_ratio = symmetric_ratio / symmetric_ratio.max()
    sparsification_threshold = np.percentile(normalized_ratio, 75)
    heuristics = np.where(normalized_ratio > sparsification_threshold, normalized_ratio, 0)
    return heuristics
def llamareevo2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    demand_ratio = np.tile(demand / capacity, (n, 1))
    capacity_ratio = np.minimum(demand_ratio, 1 - demand_ratio.T)

    # Non-linear capacity ratio with a higher power for stronger effect
    non_linear_capacity_ratio = capacity_ratio ** 8

    # Non-linear demand ratio to emphasize smaller demands
    non_linear_demand_ratio = demand_ratio ** 0.1

    # Combine various factors to determine how promising it is to select an edge
    heuristics = np.multiply(non_linear_demand_ratio, non_linear_capacity_ratio)
    heuristics = np.multiply(heuristics, (1 - demand_ratio.T) ** 4)

    # Dynamic sparsification threshold with a lower percentile for stronger sparsification
    sparsification_threshold = np.percentile(heuristics, 1)

    heuristics[heuristics < sparsification_threshold] = 0

    # Normalize the heuristics matrix
    heuristics = heuristics / np.max(heuristics)

    return heuristics
def llamareevo3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    demand_ratio = np.tile(demand / demand.max(), (n, 1))
    capacity_ratio = np.tile(demand / capacity, (n, 1))
    combined_ratio = demand_ratio * capacity_ratio

    # Item pairing strategy: prioritize pairs with complementary sizes
    pairing_scores = np.abs(combined_ratio - 0.5)
    pairing_weights = 1 - pairing_scores / pairing_scores.max()

    # Adaptive thresholding: adjust threshold based on demand distribution
    sparsification_threshold = np.percentile(combined_ratio, 25) * (1 + np.std(demand) / demand.mean())

    heuristics = np.where(combined_ratio > sparsification_threshold, combined_ratio * pairing_weights, 0)
    return heuristics


def llama3h1(demand: np.ndarray, capacity: int) -> np.ndarray:
    """
    This function calculates the heuristics for the Bin Packing Problem (BPP).

    Parameters:
    demand (np.ndarray): A 1D array representing the sizes of the items.
    capacity (int): The capacity of each bin.

    Returns:
    np.ndarray: A 2D array where heuristics[i][j] represents how promising it is to put item i and item j in the same bin.
    """

    # Calculate the complementarity of each pair of items
    # The complementarity is the difference between the capacity and the sum of the demands of the two items
    complementarity = capacity - np.add.outer(demand, demand)

    # Apply exponential decay to the complementarity values
    # This reduces the dominance of large values and emphasizes the importance of small values
    decayed_complementarity = np.exp(-complementarity / capacity)

    # Normalize the demand values to be between 0 and 1
    normalized_demand = demand / demand.max()

    # Calculate the heuristic value for each pair of items
    # The heuristic value is the product of the normalized demands and the decayed complementarity
    heuristics = np.outer(normalized_demand, normalized_demand) * decayed_complementarity

    # Sparsify the matrix by setting unpromising elements to zero
    # Here, we consider elements with a value less than 0.5 as unpromising
    heuristics[heuristics < 0.5] = 0

    return heuristics


def llama3h2(demand: np.ndarray, capacity: int) -> np.ndarray:
    """
    This function calculates the heuristics for the Bin Packing Problem (BPP).

    Parameters:
    demand (np.ndarray): A 1D array representing the sizes of the items.
    capacity (int): The capacity of each bin.

    Returns:
    np.ndarray: A 2D array where heuristics[i][j] represents how promising it is to put item i and item j in the same bin.
    """

    # Normalize the demand values to be between 0 and 1
    normalized_demand = demand / demand.max()

    # Calculate the heuristic value for each pair of items
    # The heuristic value is the product of the normalized demands of the two items
    heuristics = np.outer(normalized_demand, normalized_demand)

    # Sparsify the matrix by setting unpromising elements to zero
    # Here, we consider elements with a value less than 0.5 as unpromising
    heuristics[heuristics < 0.5] = 0

    return heuristics
def llamap3(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    demand_ratio = demand / capacity
    complementarity = np.outer(demand_ratio, demand_ratio)
    feasibility = np.outer(demand, demand) <= capacity ** 2
    sparsification_threshold = np.percentile(complementarity[feasibility], 75)
    heuristics = np.where(complementarity > sparsification_threshold, complementarity, 0)
    return heuristics


def llamap2(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]

    # Normalize demand to be between 0 and 1
    demand_normalized = demand / demand.max()

    # Calculate complementarity (how well two items fit together)
    complementarity = np.outer(demand_normalized, demand_normalized)

    # Calculate size factor (how much space two items take up together)
    size_factor = np.outer(demand, demand) / capacity ** 2

    # Combine complementarity and size factor with weights
    weights = 0.6 * complementarity + 0.4 * size_factor

    # Add controlled randomness to the weights
    randomness = np.random.rand(n, n) * 0.05
    weights += randomness

    # Remove self-similarity (don't put an item with itself)
    weights -= np.diag(np.diag(weights))

    # Threshold to emphasize top pairs (set lower weights to 0)
    threshold = np.percentile(weights, 75)
    weights[weights < threshold] = 0

    return weights
def llamap1(demand: np.ndarray, capacity: int) -> np.ndarray:
    n = demand.shape[0]
    demand_ratio = demand / capacity
    demand_ratio_matrix = np.tile(demand_ratio, (n, 1))

    # Calculate the complementarity of items
    complementarity = 1 - np.abs(demand_ratio_matrix - demand_ratio_matrix.T)

    # Enhance complementarity with high-demand differences
    demand_ratio_diff = np.abs(demand_ratio_matrix - demand_ratio_matrix.T)
    demand_ratio_diff[demand_ratio_diff < np.percentile(demand_ratio_diff, 70)] = 0
    complementarity *= demand_ratio_diff

    # Calculate the total size of each pair of items
    total_size = demand_ratio_matrix + demand_ratio_matrix.T

    # Calculate the capacity utilization for each pair of items
    capacity_utilization = 1 - np.abs(1 - total_size)

    # Enhance capacity_utilization with adaptive thresholds
    threshold = np.percentile(capacity_utilization, 80)
    capacity_utilization[capacity_utilization < threshold] = 0

    # Normalize and sparsify Interaction_matrix to focus on most promising pairs
    interaction_matrix = complementarity * capacity_utilization
    interaction_matrix = interaction_matrix / np.max(interaction_matrix)
    interaction_matrix[interaction_matrix < np.percentile(interaction_matrix, 90)] = 0

    # Simplify for direct, high-probability fits
    heuristics = interaction_matrix ** 2

    # Apply penalty for overfitting
    heuristics *= 0.9

    return heuristics


def llama3h3(demand: np.ndarray, capacity: int) -> np.ndarray:
    """
    This function calculates the heuristics for the Bin Packing Problem (BPP).

    Parameters:
    demand (np.ndarray): A 1D array representing the sizes of the items.
    capacity (int): The capacity of each bin.

    Returns:
    np.ndarray: A 2D array where heuristics[i][j] represents how promising it is to put item i and item j in the same bin.
    """

    # Normalize the demand values to be between 0 and 1
    normalized_demand = demand / demand.max()

    # Calculate the heuristic value for each pair of items
    # The heuristic value is the product of the normalized demands of the two items
    heuristics = np.outer(normalized_demand, normalized_demand)

    # Sparsify the matrix by setting unpromising elements to zero
    # Here, we consider elements with a value less than 0.5 as unpromising
    heuristics[heuristics < 0.5] = 0

    return heuristics


def llama3random(demand: np.ndarray, capacity: int) -> np.ndarray:
    return np.tile(demand/demand.max(), (demand.shape[0], 1))

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
def aco_bpp(heuristics_human):
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
    #print("random1")
    #aco_bpp(random1)
    #print("random2")
    #aco_bpp(random2)
    #print("random3")
    #aco_bpp(random3)
    #print("p1")
    #aco_bpp(p1)
    #print("p2")
    #aco_bpp(p2)
   # print("p3")
    #aco_bpp(p3)
    """print("reevo1")
    aco_bpp(reevo1)
    print("reevo2")
    aco_bpp(reevo2)
    print("reevo3")
    aco_bpp(reevo3)"""
    #print("pppp1")
    #aco_bpp(pppp1)
    #print("pppp2")
    #aco_bpp(pppp2)
    print("newp1")
    #aco_bpp(newp1)
    print("newp2")
    #aco_bpp(newp2)
    print("888855588")
    #aco_bpp(llamaeoh1)
    #aco_bpp(llamaeoh2)
    #aco_bpp(llamaeoh3)
    aco_bpp(llamap1)
    aco_bpp(llamap2)
    aco_bpp(llamap3)
    #aco_bpp(llama3random)

if __name__ == "__main__":
        main()