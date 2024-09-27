from aco import ACO
import numpy as np
import torch
import logging

import numpy as np

N_ANTS = 10
N_ITERATIONS = [1, 10, 20, 30, 40, 50]

def humman(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return prize / np.sum(weight, axis=1)

def random1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight per dimension
    avg_weight = np.mean(weight, axis=0)

    # Calculate the normalized prize to weight ratio for each item
    normalized_ratio = prize / (np.sum(weight * avg_weight, axis=1))

    # Calculate the z-score of the normalized ratio to identify outliers
    z_scores = (normalized_ratio - np.mean(normalized_ratio)) / np.std(normalized_ratio)

    # Create a mask where heuristics are set to zero if the z-score is below a threshold
    # The threshold is set to 0 for simplicity, which means we sparsify all negative z-scores
    threshold = 0
    mask = z_scores > threshold

    # Apply the mask to the normalized ratio to create the final heuristics
    heuristics = mask * normalized_ratio

    return heuristics
def random2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight per dimension
    avg_weight_per_dim = np.mean(weight, axis=0)
    # Calculate a heuristic score based on prize-to-weight ratio adjusted by average weight
    heuristic_scores = prize / (np.sum(weight * avg_weight_per_dim, axis=1) + 1e-10)  # Add a small constant to avoid division by zero
    # Set a threshold to sparsify the heuristic scores
    threshold = np.median(heuristic_scores)
    # Apply the threshold to set unpromising elements to zero
    heuristic_scores[heuristic_scores < threshold] = 0
    return heuristic_scores

def random3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight per dimension
    avg_weight_per_dim = np.mean(weight, axis=0)
    # Calculate an adjusted prize by considering the average weight
    adjusted_prize = prize / (np.sum(weight * avg_weight_per_dim, axis=1) + 1e-10)
    # Sparsify the heuristics by setting the lowest 50% to zero
    threshold = np.percentile(adjusted_prize, 50)
    heuristics = np.where(adjusted_prize > threshold, adjusted_prize, 0)
    return heuristics


def hercules1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the density of each item as the ratio of its prize to its weight sum
    total_weight = np.sum(weight, axis=1)
    density = prize / np.where(total_weight == 0, 1, total_weight)

    # Calculate the average density and prize
    avg_density = np.mean(density)
    avg_prize = np.mean(prize)

    # Adjust the density by factoring in the average density and prize
    adjusted_density = density * (density / avg_density) * (prize / avg_prize)

    # Sparsify by setting density values significantly lower than the average to zero
    sparsified_heuristics = np.where(adjusted_density < avg_density / 2, 0, adjusted_density)

    return sparsified_heuristics


def hercules2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized density
    total_weight = np.sum(weight, axis=1)
    density = prize / (total_weight + 1e-10)  # Avoid division by zero
    max_density = np.max(density)

    # Normalize the density
    normalized_density = density / max_density

    # Calculate the weight excess for each item
    weight_excess = np.maximum(0, total_weight - 1)

    # Penalize the density even more heavily according to the weight excess, with a dynamic penalty factor
    penalty_factor = 0.3  # Even higher penalty factor for excess weight
    penalized_density = normalized_density - penalty_factor * weight_excess

    # Apply an even stronger non-linear transformation to emphasize differences between high density items
    transformation_factor = 4.0  # Stronger transformation factor for emphasizing density differences
    transformed_density = np.power(penalized_density, transformation_factor)

    # Sparsify further by setting low density values to zero based on a more aggressive dynamic threshold
    threshold = np.percentile(transformed_density, 30)  # More aggressive threshold for increased sparsity
    heuristics = np.where(transformed_density >= threshold, transformed_density, 0)

    return heuristics


def hercules3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight_penalty = np.sqrt(np.sum(weight ** 2, axis=1))
    value_weight_ratio = prize / (weight_penalty + 1e-8)

    weight_violation_penalty = np.sum(np.maximum(0, weight - 1), axis=1)
    nonlinear_penalty = weight_violation_penalty ** 3

    adjusted_ratio = value_weight_ratio / (1 + nonlinear_penalty)

    median_adjusted_ratio = np.median(adjusted_ratio)
    normalized_adjusted_ratio = (adjusted_ratio - median_adjusted_ratio) / (
                np.max(adjusted_ratio) - median_adjusted_ratio + 1e-8)

    dynamic_threshold = np.median(normalized_adjusted_ratio)

    heuristics = np.where(normalized_adjusted_ratio < dynamic_threshold, 0, normalized_adjusted_ratio ** 2)

    return heuristics


def reevo1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized prize to weight ratio
    normalized_ratio = prize / np.sum(weight, axis=1)
    # Calculate the density of prizes, higher density means more promising
    density = prize / np.linalg.norm(weight, axis=1)
    # Calculate an item penalty for items that almost exceed the weight limits
    penalty = np.sum(np.clip(weight - 1, 0, None), axis=1)
    # Inverse the penalty to turn it into a reward for items that don't come close to exceeding the limits
    inverse_penalty = 1 / (1 + penalty)

    # Combine the normalized ratio, the density, and the inverse penalty to determine the heuristic score
    combined_score = normalized_ratio * density * inverse_penalty
    # Sparsify by setting the bottom 75% of scores to zero to focus on the most promising items
    threshold = np.percentile(combined_score, 25)
    heuristics = np.where(combined_score > threshold, combined_score, 0)

    return heuristics


def reevo2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate density of prize per unit weight
    density = prize / np.sum(weight, axis=1)

    # Calculate average and standard deviation of the density
    avg_density = np.mean(density)
    std_density = np.std(density)

    # Calculate score based on how many standard deviations an item's density is above the mean
    score_density = (density - avg_density) / std_density if std_density != 0 else density - avg_density

    # Incorporate a penalty for weights near the capacity (1 in each dimension)
    penalty_weight = np.sum(np.where(weight > 0.9, 1, 0), axis=1)

    # Combine density-based score with weight penalty
    score = score_density - penalty_weight

    # Normalize the score to have a positive mean and standard deviation
    avg_score = np.mean(score)
    std_score = np.std(score)
    normalized_score = (score - avg_score) / std_score if std_score != 0 else score - avg_score

    # Enforce sparsity by setting elements below a certain threshold to zero
    threshold = avg_score / 2  # setting threshold as half the average score
    heuristics = np.where(normalized_score > threshold, normalized_score, 0)

    return heuristics


def reevo3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight per dimension
    avg_weight_per_dim = np.mean(weight, axis=0)

    # Calculate density score for each item
    density_score = prize / (np.sum(weight * avg_weight_per_dim, axis=1) + 1e-8)

    # Calculate the penalty for items near or over the weight limit
    penalty = np.maximum(0, np.sum(weight, axis=1) - 1)  # Items violating the constraint get a penalty
    adjusted_density_score = density_score - penalty

    # Normalize the adjusted density score for fair comparison
    normalized_density_score = (adjusted_density_score - np.mean(adjusted_density_score)) / (
                np.std(adjusted_density_score) + 1e-8)

    # Use percentile-based threshold for sparsity
    threshold = np.percentile(normalized_density_score, 50)  # Use the median as a threshold

    # Apply sparsity to the normalized score
    sparse_density_score = np.where(normalized_density_score < threshold, 0, normalized_density_score)

    return sparse_density_score
def gpt(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate average prize over each item
    avg_prize_per_weight = prize / np.linalg.norm(weight, axis=1)

    # Applying a scalar factor based on weight
    weight_factor = np.exp(-np.sum(weight, axis=1))  # Diminish value relative to weight

    # Compute overall heuristic by multiplying both factors
    heuristic_scores = avg_prize_per_weight * weight_factor

    # Setting low scoring elements to zero for sparsification of the heuristic
    heuristic_scores[heuristic_scores < np.mean(heuristic_scores)] = 0

    return heuristic_scores

def hhh1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight per dimension
    avg_weight = np.mean(weight, axis=0)

    # Adaptive penalty term based on the capacity constraint violation
    penalties = np.maximum(0, np.sum(weight, axis=1) - 1)
    penalties = 1 / (1 + penalties)

    # Adjusting the profitability calculation with penalties and average weight
    profitability = prize * penalties / (np.sum(weight * avg_weight, axis=1) + 1e-10)

    # Use the weighted median as a threshold to balance exploration and exploitation
    median_profitability = np.median(profitability)

    # Heuristic scores start as all zeros
    heuristics = np.zeros_like(profitability)

    # Assign heuristic score based on profitability compared to the median
    heuristics[profitability >= median_profitability] = profitability[profitability >= median_profitability]

    return heuristics


def hhh2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the density of each item
    total_weight = np.sum(weight, axis=1)
    density = prize / total_weight

    # Calculate the average density
    avg_density = np.mean(density)

    # Penalize items with density below a threshold (e.g., 70% of the average density)
    threshold = 0.7 * avg_density

    # Calculate a penalization factor based on how far below the threshold the density is
    penalization_factor = np.where(density < threshold, 1 - (density / threshold), 1)

    # Calculate a score considering both density and penalization factor
    score = density * penalization_factor

    # Calculate weight sparsity by counting the number of dimensions with weight > 0 for each item
    sparsity = np.sum(weight > 0, axis=1)

    # Promote diversity by inversely proportional sparsity
    diversity_factor = 1 / (1 + sparsity)

    # Adjust the heuristic scores with diversity promotion
    heuristic_scores = score * diversity_factor

    # Normalize scores to maintain a comparable scale for different problems
    heuristic_scores = (heuristic_scores - np.mean(heuristic_scores)) / np.std(heuristic_scores)

    # Set heuristic scores of unpromising elements (e.g., negative scores) to zero
    heuristic_scores = np.where(heuristic_scores > 0, heuristic_scores, 0)

    return heuristic_scores


def hhh3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Normalize the weight measure considering the maximum weight constraint
    normalized_weight_measure = weight / np.max(np.sum(weight ** 2, axis=1) ** 0.5)

    # Prize-to-weight efficiency as the measure of item value
    prize_to_weight_efficiency = prize / np.sum(normalized_weight_measure ** 2, axis=1) ** 0.5

    # Introduce a sparsity factor with an adaptive exponent
    max_efficiency = np.max(prize_to_weight_efficiency)
    sparsity_exponent = 2  # Adjusted exponent for balancing sparsity
    sparsity_factor = np.exp(-0.5 * ((prize_to_weight_efficiency / max_efficiency) ** sparsity_exponent))

    # Generate heuristic scores with adjusted sparsity
    heuristic_scores = prize_to_weight_efficiency * (1 - sparsity_factor)

    # Set the threshold adaptively based on the standard deviation of heuristic scores
    std_dev = np.std(heuristic_scores)
    threshold = np.mean(heuristic_scores) - std_dev

    # Apply the threshold to enforce sparsity and filter out less promising items
    heuristic_scores[heuristic_scores < threshold] = 0

    # Compute the diversity penalty to avoid selecting similar items
    pairwise_distances = np.mean(
        [np.exp(-np.sum((weight - weight[i]) ** 2, axis=1) ** 0.5) for i in range(len(weight))], axis=0)
    diversity_penalty = 1 - pairwise_distances

    # Adjust the heuristic scores by the diversity penalty
    heuristic_scores *= diversity_penalty

    # Normalize the heuristic scores to ensure they are comparable
    heuristic_scores = (heuristic_scores - np.min(heuristic_scores)) / (
                np.max(heuristic_scores) - np.min(heuristic_scores))

    return heuristic_scores


def new1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate item density
    density = prize / np.sum(weight, axis=1)

    # Calculate non-linear penalty for weight constraint violations
    penalty_factor = np.maximum(0, np.sum(weight, axis=1) - 1)
    penalty_factor = np.exp(penalty_factor)  # Use exponential penalty

    # Soft penalty for exceeding weight constraints
    soft_penalty = 1 / (1 + penalty_factor)

    # Calculate diversity and rarity
    diversity = 1 / (1 + np.sum(weight >= 0.5, axis=1))
    rarity = np.sum(density >= density[:, np.newaxis], axis=1)
    rarity_factor = diversity * (1 / (1 + rarity))

    # Combine density, soft penalty, rarity_factor to compute raw heuristics
    raw_heuristics = density * soft_penalty * rarity_factor

    # Apply dynamic threshold based on the median to sparsify
    dynamic_threshold = np.median(raw_heuristics)
    heuristics_v2 = np.where(raw_heuristics >= dynamic_threshold, raw_heuristics, 0)

    # Normalize the heuristics to ensure robustness and stability
    heuristics_v2_normalized = heuristics_v2 / np.sum(heuristics_v2)

    return heuristics_v2_normalized
def new2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Prize-to-weight density score
    density = prize / np.sum(weight, axis=1)

    # Non-linear penalty based on the sum of squared weights raised to an even higher power
    weight_penalty = 1 / (1 + np.sum(weight ** 4, axis=1) ** 0.25)

    # Enhanced deviation score that more heavily penalizes common weight distributions
    deviation = 1 / (np.std(weight, axis=1) + 1e-8)  # Adding a small constant to avoid division by zero

    # Combine scores with an updated formula prioritizing density and penalizing commonality
    composite_score = (density ** 3) * weight_penalty * deviation

    # Normalize the scores to bring them within a comparable range to each other
    normalized_composite_score = composite_score / np.max(composite_score)

    # Apply a more aggressive sparsity threshold based on a lower quantile to filter out less promising items
    threshold = np.quantile(normalized_composite_score, 0.25)
    heuristics = np.where(normalized_composite_score > threshold, normalized_composite_score, 0)

    return heuristics
def new3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized prize-to-weight ratio for each item
    weight_sum = np.sum(weight, axis=1)
    normalized_ratio = prize / weight_sum

    # Calculate a softer penalty for weight constraint violations
    penalty_factor = np.sum(weight > 1, axis=1)
    penalty = 1 - 0.1 * penalty_factor

    # Calculate an adjusted density by incorporating the square of the weight
    adjusted_density = prize * (1 / np.sum(weight**2, axis=1))

    # Combine the normalized ratio, penalty, and adjusted density to create a heuristic score
    heuristic_score = normalized_ratio * penalty * adjusted_density

    # Use a gradient approach instead of binary thresholding
    # Assume that a lower normalized ratio should decrease the heuristic score gradually
    gradient_score = np.log(1 + normalized_ratio)
    heuristic_score = heuristic_score * gradient_score

    return heuristic_score


def newp1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight per prize unit for each item
    avg_weight_per_prize = np.sum(weight, axis=1) / prize

    # Calculate the density of each item as the ratio of prize to total weight
    density = prize / np.sum(weight, axis=1)

    # Adjust the density by the average weight per prize unit
    adjusted_density = density / avg_weight_per_prize

    # Sparsify the heuristic values by setting those below a certain threshold to zero
    threshold = np.median(adjusted_density)
    heuristics = np.where(adjusted_density >= threshold, adjusted_density, 0)

    return heuristics


def newp2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the value-to-weight density for each item
    density = prize / np.sum(weight, axis=1)

    # Penalize items that violate any weight dimension's capacity
    penalty_factor = np.prod(np.where(weight > 1, 0, 1), axis=1)

    # Apply penalty factor to density to obtain penalized density
    penalized_density = density * penalty_factor

    # Boost densities of non-penalized items using a non-linear transformation to highlight the most promising items
    non_linear_boost = np.power(penalized_density[penalized_density > 0], 1.5)

    # Combine penalized densities with non-linear boosts
    combined_density = np.where(penalty_factor > 0, non_linear_boost, penalized_density)

    # Calculate the average penalized and boosted density for normalization
    avg_combined_density = np.mean(combined_density[combined_density > 0])

    # Boost the densities that are above average and penalize the ones below average
    final_density = np.where(combined_density >= avg_combined_density, combined_density ** 1.1, combined_density / 1.1)

    # Normalize the final densities to bring all values between 0 and 1
    heuristics = (final_density - np.min(final_density)) / (np.max(final_density) - np.min(final_density) + 1e-10)

    return heuristics


def newp3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the average weight per dimension
    avg_weight = np.mean(weight, axis=0)

    # Calculate the normalized prize-to-weight ratio
    normalized_ratio = prize / (
                np.sum(weight * avg_weight, axis=1) + 1e-6)  # adding a small constant to avoid division by zero

    # Sparsify the heuristic by setting elements below a certain percentile to zero
    threshold = np.percentile(normalized_ratio, 50)  # setting threshold to 50th percentile
    heuristic = np.where(normalized_ratio >= threshold, normalized_ratio, 0)

    return heuristic


def hpp1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight_sum = np.sum(weight, axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        density = prize / (weight_sum + 1e-8)
        density[weight_sum == 0] = 0

    penalty_factor = np.maximum(0, weight - 1).sum(axis=1)
    penalized_density = density * np.exp(100 * penalty_factor)

    sparsity_factor = np.sum(density >= density[:, None], axis=1)
    sparsity_adjustment = 1 / (sparsity_factor + 1)  # Adjust to avoid division by zero

    adjusted_density = penalized_density * sparsity_adjustment

    # Introduce Gaussian noise prudently for exploration
    noise = np.random.normal(0, 0.001, adjusted_density.shape)
    heuristic_scores = adjusted_density + noise

    # Normalize the heuristic scores more robustly
    heuristic_scores -= np.min(heuristic_scores)
    heuristic_scores /= (np.max(heuristic_scores) + 1e-8)

    # Zero out low scores to simplify choices
    heuristic_scores[heuristic_scores < np.percentile(heuristic_scores, 20)] = 0

    return heuristic_scores


def hpp2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate density as the prize-to-weight ratio normalized over L2 norm of weight
    density = prize / np.linalg.norm(weight, axis=1)

    # Calculate penalty term based on how much each item exceeds the weight limit in each dimension
    penalty = np.maximum(0, np.sum(weight, axis=1) - 1)

    # Calculate the adjusted score by subtracting penalty from density
    adjusted_scores = density - penalty

    # Calculate robust stats for normalization
    median_score = np.median(adjusted_scores)
    mad_score = np.median(np.abs(adjusted_scores - median_score))  # Median absolute deviation

    # Normalize scores using robust measures
    normalized_scores = (adjusted_scores - median_score) / mad_score

    # Sparsify heuristic by zeroing out elements below a certain threshold
    # Here, we use the median score as the threshold to filter out less promising items
    heuristic_scores = np.where(normalized_scores > 0, normalized_scores, 0)

    return heuristic_scores


def hpp3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight_sum = np.sum(weight, axis=1)
    density = prize / weight_sum  # Value-to-weight density

    # Calculate normalized weight and use it to compute the non-linearity factor
    normalized_weight = weight / weight_sum[:, None]
    non_linearity_factor = np.prod(1 - normalized_weight, axis=1)

    # Calculate the interdependency factor based on the average overlap with other items
    interdependency_matrix = np.dot(normalized_weight, normalized_weight.T)
    np.fill_diagonal(interdependency_matrix, 0)  # Exclude self-overlap
    interdependency_factor = np.mean(interdependency_matrix, axis=1)

    # Inverse interdependency to make it a factor that promotes diversity
    interdependency_factor = 1.0 / (1.0 + interdependency_factor)

    # Combine density with non-linearity and inverse interdependency factors
    heuristic_scores = density * non_linearity_factor * interdependency_factor

    # Sparsify by setting heuristic scores of the less promising items to zero
    # Use a dynamic threshold based on the mean and standard deviation
    mean_scores = np.mean(heuristic_scores)
    std_scores = np.std(heuristic_scores)
    dynamic_threshold = mean_scores - 0.5 * std_scores

    # Apply the dynamic threshold to maintain diversity
    heuristic_scores[heuristic_scores < dynamic_threshold] = 0

    return heuristic_scores


def eoh1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate a density score for each item as the ratio of its prize to the maximum weight dimension
    density = prize / np.max(weight, axis=1)

    # Calculate a penalty factor based on the sum of weights exceeding the limit in each dimension
    penalty_factor = np.sum(weight > 1, axis=1)

    # Adjust the density score with the penalty factor
    adjusted_density = density / (1 + penalty_factor)

    # Use a threshold to create a binary heuristic score
    threshold = np.mean(adjusted_density)
    heuristic_score = np.where(adjusted_density >= threshold, adjusted_density, 0)

    # Normalize the heuristic scores to make them more comparable to the original heuristic scores
    max_score = np.max(heuristic_score)
    heuristic_score = heuristic_score / max_score if max_score > 0 else heuristic_score

    return heuristic_score
def eoh2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the density of each item
    density = prize / np.sum(weight, axis=1)
    # Find the density quantiles to create a threshold
    quantiles = np.quantile(density, [0.25, 0.5, 0.75])
    # Calculate the difference between item density and median density
    diff_density = density - quantiles[1]
    # Use the interquartile range to normalize the difference
    iqr = quantiles[2] - quantiles[0]
    normalized_diff = diff_density / iqr if iqr != 0 else diff_density
    # Sparsify based on normalized difference: keep high values and discard low or negative ones
    heuristics = np.where(normalized_diff > 0, normalized_diff, 0)
    return heuristics
def eoh3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized prize to weight ratio for each item
    normalized_ratio = prize / np.sum(weight**2, axis=1)**0.5
    # Calculate the standard deviation of the normalized ratio
    std_normalized_ratio = np.std(normalized_ratio)
    # Calculate the mean of the normalized ratio
    mean_normalized_ratio = np.mean(normalized_ratio)
    # Adjust the heuristic score calculation to give more weight to items
    # with significantly higher or lower ratios compared to the mean
    heuristic_scores = 1.5 * (normalized_ratio - mean_normalized_ratio) / (std_normalized_ratio + 1e-8) + normalized_ratio
    # Modify the percentile for sparsification and set a higher percentile of low scores to zero
    threshold = np.percentile(heuristic_scores, 30)
    heuristic_scores[heuristic_scores < threshold] = 0
    return heuristic_scores


def llamaeoh1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]

    # Calculate the prize-to-weight ratio for each item
    ratio = prize / np.sum(weight, axis=1)

    # Calculate the average weight of each item across all dimensions
    avg_weight = np.mean(weight, axis=1)

    # Calculate the standard deviation of weights for each item across all dimensions
    std_weight = np.std(weight, axis=1)

    # Calculate the heuristic value for each item
    heuristics = ratio * (1 - avg_weight) * (1 + std_weight)

    # Sparsify the heuristics by setting unpromising elements to zero
    threshold = np.percentile(heuristics, 50)  # adjust the threshold as needed
    heuristics[heuristics < threshold] = 0

    return heuristics


def llamaeoh2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]

    # Calculate the prize-to-weight ratio for each item
    ratio = prize / np.sum(weight, axis=1)

    # Calculate the average weight of each item across all dimensions
    avg_weight = np.mean(weight, axis=1)

    # Calculate the standard deviation of weights for each item across all dimensions
    std_weight = np.std(weight, axis=1)

    # Calculate the heuristic value for each item
    heuristics = ratio * (1 - avg_weight) * (1 + std_weight)

    # Sparsify the heuristics by setting unpromising elements to zero
    threshold = np.percentile(heuristics, 50)  # adjust the threshold as needed
    heuristics[heuristics < threshold] = 0

    return heuristics


def llamaeoh3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    n, m = prize.shape[0], weight.shape[1]

    # Calculate the prize-to-weight ratio for each item
    ratio = prize / np.sum(weight, axis=1)

    # Calculate the average weight of each item across all dimensions
    avg_weight = np.mean(weight, axis=1)

    # Calculate the standard deviation of weights for each item across all dimensions
    std_weight = np.std(weight, axis=1)

    # Calculate the heuristic value for each item
    heuristics = ratio * (1 - avg_weight) * (1 + std_weight)

    # Sparsify the heuristics by setting unpromising elements to zero
    threshold = np.percentile(heuristics, 50)  # adjust the threshold as needed
    heuristics[heuristics < threshold] = 0

    return heuristics


def pp1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized density of each item
    total_weight = np.sum(weight, axis=1)
    normalized_density = prize / (total_weight + 1e-8)  # Adding a small constant to avoid division by zero

    # Calculate a light penalty for weight constraint violation
    penalty_factor = np.maximum(0, weight - 1).sum(axis=1)
    light_penalty = 1 + 0.1 * penalty_factor  # Apply a light penalty for violated constraints

    # Adjust density with light penalty
    adjusted_density = normalized_density / light_penalty

    # Calculate the maximum adjusted density to use as a scaling factor
    max_adjusted_density = np.max(adjusted_density)

    # Use a power transformation to enhance the contrast between high and low density items
    heuristic_values = np.power(adjusted_density / max_adjusted_density, 3)

    return heuristic_values


def pp2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate density as the ratio of prize to the norm of the weight vector
    density = prize / np.linalg.norm(weight, axis=1)

    # Calculate space left as the minimum space left in any dimension for each item
    space_left = np.min(1 - weight, axis=1)

    # Calculate a penalty factor based on the lack of space
    penalty_factor = np.maximum(0, 1 - space_left)

    # Adjust the density by the penalty factor, encouraging items with more space left
    adjusted_density = density * (1 - penalty_factor)

    # Use a combination of mean and median as a heuristic threshold to balance exploration and exploitation
    threshold = 0.5 * (np.mean(adjusted_density) + np.median(adjusted_density))

    # Create the heuristics by penalizing items below the threshold
    heuristics = np.where(adjusted_density >= threshold, adjusted_density, 0)

    return heuristics

def gemmaeoh1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    utility = prize / np.sum(weight, axis=1)
    density = prize / (np.max(weight, axis=1) + 1e-9)
    heuristics = utility * density
    threshold = np.percentile(heuristics, 20)  # Sparsify by keeping top 20%
    heuristics[heuristics < threshold] = 0
    return heuristics
def gemmaeoh2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_density = prize / np.max(weight, axis=1)
    heuristics = np.exp(0.5 * prize_density)
    return heuristics
def gemmaeoh3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weighted_prize = prize / np.max(weight, axis=1)
    density = np.sum(weight, axis=1)
    heuristics = weighted_prize * np.exp(-density * 0.8)
    threshold = np.percentile(heuristics, 50)
    heuristics[heuristics < threshold] = 0
    return heuristics
def gemmarandom1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    max_weight_ratio = np.max(weight, axis=1) / np.sum(weight, axis=1)
    heuristics = prize_per_unit_weight * (1 - max_weight_ratio)
    threshold = np.percentile(heuristics, 50)  # You can adjust the percentile
    heuristics[heuristics < threshold] = 0
    return heuristics
def gemmarandom2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    price_per_unit_weight = prize / np.sum(weight, axis=1)
    max_weight = np.max(weight, axis=1)
    heuristics = price_per_unit_weight * (1.0 / (max_weight + 0.1)) # Adjust 0.1 for sparsity
    heuristics[heuristics < np.mean(heuristics) * 0.8] = 0 # Sparsify
    return heuristics
def gemmarandom3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    max_weight_ratio = np.max(weight, axis=1) / np.sum(weight, axis=1)
    heuristics = prize_per_unit_weight * (1 - max_weight_ratio)
    threshold = np.percentile(heuristics, 50)  # Adjust percentile as needed
    heuristics[heuristics < threshold] = 0
    return heuristics


def gemmareevo1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    weight_balance = np.max(weight, axis=1) / np.sum(weight, axis=1)
    heuristics = prize_per_unit_weight * (1 - weight_balance)

    # Sparsify the heuristics
    threshold = np.mean(heuristics)
    heuristics[heuristics < threshold] = 0

    return heuristics
def gemmareevo2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    dimension_weights = np.linspace(0.8, 1.2, weight.shape[1])
    penalty = np.sum(weight**2 * dimension_weights, axis=1) * np.exp(0.5 * np.sum(weight * dimension_weights, axis=1))
    density = prize / np.sum(weight, axis=1)
    interaction_penalty = np.sum(weight[:, None] * weight[None, :], axis=2)
    heuristics = prize / (penalty + 1e-6) ** 0.7 * (1 + density) * np.exp(-0.05 * interaction_penalty.mean(axis=1))
    threshold = np.percentile(heuristics, 30)
    heuristics[heuristics < threshold] = 0
    return heuristics
def gemmareevo3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    max_weight_ratio = np.max(weight, axis=1) / np.sum(weight, axis=1)
    # Non-linear penalty for exceeding average weights
    weight_penalty = np.exp(np.sum(weight, axis=1) - 1)
    # Penalize dimensional imbalances
    dim_imbalance_penalty = np.std(weight, axis=1)
    heuristics = prize_per_unit_weight * (1 - max_weight_ratio) / (weight_penalty * (1 + dim_imbalance_penalty))
    threshold = np.percentile(heuristics, 50)
    heuristics[heuristics < threshold] = 0
    return heuristics
def gemmah1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    weight_penalty = np.max(weight, axis=1)
    heuristics = prize_per_unit_weight / weight_penalty
    threshold = np.percentile(heuristics, 20)
    heuristics[heuristics < threshold] = 0
    return heuristics


def gemmah2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    max_weight_ratios = np.max(weight / np.expand_dims(np.sum(weight, axis=1), axis=1), axis=1)
    density_score = prize_per_unit_weight * (1 - max_weight_ratios)

    # Weight Magnitude Awareness
    weight_magnitude = np.sum(weight, axis=1)
    magnitude_bonus = np.exp(-weight_magnitude / np.max(weight_magnitude))

    # Distribution Awareness with Adaptive IQR
    density_percentile_75 = np.percentile(density_score, 75)
    density_percentile_25 = np.percentile(density_score, 25)
    iqr = density_percentile_75 - density_percentile_25
    adaptive_iqr_window = 0.3 * iqr
    distribution_factor = np.where(density_score > density_percentile_75, 1.2,
                                   np.where(density_score > density_percentile_75 - adaptive_iqr_window, 1, 0.5))

    # Dimensionality-Weighted Density Scores (Tighter Coupling and Exponent Tuning)
    dimensionality_weights = np.sum(weight > 0, axis=1) / weight.shape[1]
    dimensionality_bonus = density_score ** (1 + dimensionality_weights * 2)

    # Sparsity Penalty
    sparsity_penalty = np.where(np.sum(weight > 0, axis=1) < weight.shape[1], 1.2, 1)

    heuristics = density_score * magnitude_bonus * distribution_factor * dimensionality_bonus * sparsity_penalty
    heuristics[heuristics < np.percentile(heuristics, 5)] = 0

    return heuristics


def gemmah3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weighted_prize = prize / np.sum(weight, axis=1)
    density = prize / np.sum(weight, axis=1)

    # Adaptive, dimension-specific penalties based on constraint ratios
    constraint_ratios = np.sum(weight, axis=0) / weight.shape[0]
    penalties = np.prod(1.0 / (1 + constraint_ratios)) ** (1 / weight.shape[1])

    # Consider weight imbalance across dimensions for each item
    imbalance = np.std(weight, axis=1) / np.mean(weight, axis=1)

    # Emphasize items with higher density and lower imbalance
    heuristics = weighted_prize * density * penalties * (1.0 / (1 + imbalance)) ** 2

    # Dynamic thresholding based on constraint tightness
    tightness = np.sum(constraint_ratios * (constraint_ratios > 0.9))
    threshold = np.percentile(heuristics[heuristics > 0], max(50 - tightness * 20, 10)) * np.random.uniform(0.6, 0.9)
    heuristics = np.where(heuristics > threshold, heuristics, 0)
    return heuristics

def gemmap1(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    std_weights = np.std(weight, axis=0)
    dim_importance = std_weights / np.sum(std_weights)

    # Non-linear score scaling
    prize_per_unit_weight = np.power(prize_per_unit_weight, 1.2)

    penalty = np.exp(np.dot(weight, dim_importance) ** 2)
    heuristics = prize_per_unit_weight / penalty

    # Diversification penalty with emphasis on high-weight dimensions
    total_weight_per_item = np.sum(weight, axis=1)
    heuristics = heuristics / (total_weight_per_item + 1e-5)
    dimension_weight = 1 / (std_weights + 1e-5)
    heuristics = heuristics * np.dot(weight, dimension_weight)

    # Adaptive thresholding
    threshold = np.percentile(heuristics, 50) + 0.2 * np.std(heuristics) + 0.1 * np.random.rand()

    # Sparsity with increased threshold
    heuristics = np.where(heuristics > threshold, heuristics, 0)
    return heuristics
def gemmap2(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    prize_per_unit_weight = prize / np.sum(weight, axis=1)
    density = np.percentile(weight, 95, axis=1)
    combined_score = prize_per_unit_weight * np.exp(-density / np.max(density)) * (1.0 / (1.0 + density))
    thresholds = np.percentile(combined_score, [20, 40, 60, 80])
    interaction_penalty = np.sum(np.sort(weight, axis=1)[:, -2:], axis=1)
    heuristics = np.where(combined_score >= thresholds[3], combined_score - 0.2*interaction_penalty,
                        np.where(combined_score >= thresholds[2], combined_score * 0.8 - 0.1*interaction_penalty,
                                 np.where(combined_score >= thresholds[1], combined_score * 0.6 - 0.05*interaction_penalty,
                                         np.where(combined_score >= thresholds[0], combined_score * 0.4 - 0.01*interaction_penalty, 0))))
    return heuristics
def gemmap3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    heuristics = np.zeros_like(prize)
    for i in range(len(prize)):
        if np.all(weight[i] <= 0.33):
            heuristics[i] = prize[i] / np.max(weight[i])
        elif np.all(weight[i] <= 0.66):
            heuristics[i] = prize[i] / np.sum(weight[i])
        else:
            heuristics[i] = prize[i] / (np.sum(weight[i]) * 1.5)
    return heuristics

def pp3(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # Calculate the normalized prize-to-weight ratio
    normalized_ratio = prize / np.sum(weight, axis=1)

    # Calculate the density of each item
    item_density = prize / np.sqrt(np.sum(weight ** 2, axis=1))

    # Calculate a penalty based on the amount each item exceeds the weight limit in each dimension
    penalty = np.sum(np.maximum(0, weight - 1), axis=1)

    # Boost the density for items that do not exceed the weight limit
    boost_factor = np.where(penalty == 0, 1.5, 1)
    boosted_density = item_density * boost_factor

    # The penalty influences the score inversely proportional to its magnitude with an adaptive threshold
    adaptive_penalty = 1 / (1 + penalty ** 1.5)

    # Combine the normalized ratio, boosted density, and adaptive penalty to determine the overall score for each item
    combined_score = normalized_ratio * boosted_density * adaptive_penalty

    # Aggressively sparsify the heuristics by setting the lowest 20% of the combined scores to zero
    threshold = np.percentile(combined_score, 20)
    heuristics = np.where(combined_score >= threshold, combined_score, 0)

    # Apply strategic boosting for top candidates
    top_candidates = np.argsort(combined_score)[-len(combined_score) // 5:]
    heuristics[top_candidates] *= 1.1

    return heuristics


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
    print("gemmareevo3")
    #mkp_aco(newp1)
    #print("hhh2")
    #mkp_aco(newp2)
    #print("hhh3")
    #mkp_aco(pp1)
    #mkp_aco(pp2)
    #mkp_aco(gemmaeoh1)
    #mkp_aco(gemmaeoh2)
    #mkp_aco(gemmaeoh3)
    #mkp_aco(gemmarandom1)
    #mkp_aco(gemmarandom2)
    #mkp_aco(gemmarandom3)
    #mkp_aco(gemmareevo1)
    #mkp_aco(gemmareevo2)
    mkp_aco(gemmareevo3)
    #mkp_aco(gemmah1)
    #mkp_aco(gemmah2)
    #mkp_aco(gemmah3)
    #mkp_aco(gemmap1)
    #mkp_aco(gemmap2)
    #mkp_aco(gemmap3)
    """print("humman")
    mkp_aco(humman)
    print("random1")
    mkp_aco(random1)
    print("random2")
    mkp_aco(random2)
    print("random3")
    mkp_aco(random3)
    print("p1")
    mkp_aco(p1)
    print("p2")
    mkp_aco(p2)
    print("p3")
    mkp_aco(p3)
    print("reevo1")
    mkp_aco(reevo1)
    print("reevo2")
    mkp_aco(reevo2)
    print("reevo3")
    mkp_aco(reevo3)
    print("hercules1")
    mkp_aco(hercules1)
    print("hercules2")
    mkp_aco(hercules2)
    print("hercules3")
    mkp_aco(hercules3)"""


if __name__ == "__main__":
        main()