import json
import numpy as np
import os


def load_data(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file '{json_file_path}' not found in the current directory.")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None


def model_normal_distribution(data):
    print("\n--- Model 1: Normal Distribution ---")
    all_amounts = [amount for round_sample in data for amount in round_sample]
    sample_mean = np.mean(all_amounts)
    sample_std = np.std(all_amounts, ddof=0)
    mu = sample_mean
    sigma = sample_std
    print(f"Estimated Mean (μ): {mu:.4f}")
    print(f"Estimated Std Dev (σ): {sigma:.4f}")
    print(f"Model: X ～ N({mu:.4f}, {sigma:.4f}^2)")
    sample_from_model = np.round(np.random.normal(loc=mu, scale=sigma, size=6), 2)
    print(f"Example sample from model (probably does not sum to total value): {sample_from_model.tolist()}")
    return mu, sigma


def estimate_total_amount(data):
    totals_per_round = [sum(round_sample) for round_sample in data]
    estimated_total = np.mean(totals_per_round)
    return estimated_total


def generate_uniform_round_v2(total_amount=50.0):
    amounts = []
    remaining_amount = total_amount
    remaining_people = 6

    for i in range(5):
        remaining_avg = remaining_amount / remaining_people
        upper_bound = min(2 * remaining_avg, remaining_amount)
        lower_bound = 0.01

        new_amount = np.random.uniform(low=lower_bound, high=upper_bound)
        new_amount = min(new_amount, remaining_amount - (remaining_people - 1) * 0.01)

        amounts.append(round(float(new_amount), 2))
        remaining_amount -= new_amount
        remaining_people -= 1

        if remaining_amount < remaining_people * 0.01:
             print("Warning: Generated amounts may not allow for minimum 0.01 for remaining people.")
             break

    if len(amounts) == 5:
        last_amount = round(float(remaining_amount), 2)
        amounts.append(last_amount)
    else:
        while len(amounts) < 6:
            amounts.append(0.0)

    return amounts


def model_uniform_distribution(data):
    print("\n--- Model 2: Uniform Distribution ---")
    estimated_total = estimate_total_amount(data)
    print(f"Estimated Total Amount per Round: {estimated_total:.2f}")
    print(f"Model Rule: For each of the first 5 amounts in a round:")
    print(f"  Amount_i ～ U(0.01, min(2 * RemainingAvg, RemainingAmount))")
    print(f"  The 6th amount is set to Total_Amount - Sum_of_first_5_amounts")
    print(f"Using Estimated Total Amount: {estimated_total:.2f}")
    generated_round = generate_uniform_round_v2(total_amount=estimated_total)
    print(f"Example generated round from model: {generated_round}")
    print(f"Sum of generated round: {sum(generated_round):.2f}")
    return estimated_total


if __name__ == "__main__":
    samples_data = load_data("Database/Wechat_Samples.json")
    if samples_data is None:
        print("Failed to load data. Exiting.")
        exit()

    mu, sigma = model_normal_distribution(samples_data)
    total_amt = model_uniform_distribution(samples_data)

    print("\n--- Summary ---")
    print(f"Model 1 (Normal Distribution): X ~ N(mean={mu:.4f}, std={sigma:.4f})")
    print(f"Model 2 (Uniform Distribution): Xi ~ Unif(0.01, 2*(S-Si)/n-i")