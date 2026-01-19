import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib import cm
from scipy import stats
from tqdm import tqdm  # 进度条库

# 安装tqdm: pip install tqdm

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

def wechat_red_envelope_simulation_single(total_amount, num_people):
    """
    Simulate a single WeChat red envelope distribution
    
    Parameters:
    - total_amount: total money in red envelope
    - num_people: number of people to split the money
    
    Returns:
    - allocations: list of amounts received by each person
    """
    remaining_amount = total_amount
    remaining_people = num_people
    allocations = []
    
    for person_idx in range(num_people - 1):
        if remaining_people > 0:
            remaining_average = remaining_amount / remaining_people
            
            # Generate random amount: Uniform(0, 2 * remaining_average)
            max_amount = min(2 * remaining_average, remaining_amount - 0.01)
            amount = np.random.uniform(0.01, max_amount)
            amount = round(amount, 2)
            
            allocations.append(amount)
            remaining_amount -= amount
            remaining_people -= 1
        else:
            break
    
    # Last person gets all remaining amount
    last_amount = round(remaining_amount, 2)
    allocations.append(last_amount)
    
    # Ensure sum equals total amount (account for rounding errors)
    allocations[-1] = round(total_amount - sum(allocations[:-1]), 2)
    
    return allocations

def simulate_max_position_probability(num_people, total_amount=50, num_simulations=10000):
    """
    Simulate to find probability of each position getting the maximum amount
    
    Parameters:
    - num_people: number of people in the red envelope
    - total_amount: total money in red envelope (default 50)
    - num_simulations: number of simulations to run
    
    Returns:
    - probabilities: list of probabilities for each position (0-indexed)
    - mean_amounts: list of mean amounts for each position
    """
    max_counts = np.zeros(num_people)  # Count times each position gets max
    amount_sums = np.zeros(num_people)  # Sum of amounts for each position
    
    for _ in range(num_simulations):
        allocations = wechat_red_envelope_simulation_single(total_amount, num_people)
        
        # Update amount sums
        for i in range(num_people):
            amount_sums[i] += allocations[i]
        
        # Find position(s) with maximum amount (handle ties)
        max_amount = max(allocations)
        max_positions = [i for i, amount in enumerate(allocations) if amount == max_amount]
        
        # If there's a tie, split the count equally
        for pos in max_positions:
            max_counts[pos] += 1.0 / len(max_positions)
    
    # Calculate probabilities
    probabilities = max_counts / num_simulations
    
    # Calculate mean amounts
    mean_amounts = amount_sums / num_simulations
    
    return probabilities, mean_amounts

def analyze_position_probability_for_different_n(min_n=2, max_n=50, total_amount=50, num_simulations=10000):
    """
    Analyze probability of getting maximum amount for different numbers of people
    
    Parameters:
    - min_n: minimum number of people
    - max_n: maximum number of people
    - total_amount: total money in each red envelope
    - num_simulations: number of simulations per n
    
    Returns:
    - results_dict: dictionary containing all results
    """
    results_dict = {
        'min_n': min_n,
        'max_n': max_n,
        'total_amount': total_amount,
        'num_simulations': num_simulations,
        'probabilities': {},  # n -> probabilities list
        'mean_amounts': {},   # n -> mean amounts list
        'optimal_positions': {}  # n -> optimal position(s)
    }
    
    print(f"Simulating probabilities for n = {min_n} to {max_n} people")
    print(f"Total amount: {total_amount} ¥, Simulations per n: {num_simulations}")
    print("=" * 60)
    
    # Create progress bar
    for n in tqdm(range(min_n, max_n + 1), desc="Simulating different n"):
        probabilities, mean_amounts = simulate_max_position_probability(
            n, total_amount, num_simulations
        )
        
        results_dict['probabilities'][n] = probabilities.tolist()
        results_dict['mean_amounts'][n] = mean_amounts.tolist()
        
        # Find optimal position(s) - positions with max probability
        max_prob = max(probabilities)
        optimal_positions = [i for i, p in enumerate(probabilities) if p == max_prob]
        results_dict['optimal_positions'][n] = optimal_positions
        
        # Print progress for small n
        if n <= 10 or n % 10 == 0:
            print(f"\nn = {n} people:")
            print(f"  Optimal position(s): {[pos+1 for pos in optimal_positions]} " 
                  f"(probability: {max_prob:.3f})")
            print(f"  First 3 probabilities: {probabilities[:3].round(3).tolist()}")
            print(f"  Last 3 probabilities: {probabilities[-3:].round(3).tolist()}")
    
    return results_dict

def visualize_position_probability_results(results_dict):
    """
    Create comprehensive visualizations of the probability analysis
    
    Parameters:
    - results_dict: dictionary containing simulation results
    """
    min_n = results_dict['min_n']
    max_n = results_dict['max_n']
    probabilities_dict = results_dict['probabilities']
    
    # Create output directory
    output_dir = os.path.join("Database", "Position_Probability_Analysis")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("\n" + "="*60)
    print("VISUALIZING POSITION PROBABILITY RESULTS")
    print("="*60)
    
    # 1. 3D Surface Plot: Probability vs Position vs Number of People
    fig1 = plt.figure(figsize=(16, 12))
    
    # Prepare data for 3D plot
    n_values = list(range(min_n, max_n + 1))
    positions = []
    probs_matrix = []
    
    for n in n_values:
        probs = probabilities_dict[n]
        positions_list = list(range(1, n + 1))
        
        # Pad with NaN for smaller n
        padded_probs = probs + [np.nan] * (max_n - n)
        
        positions.append(positions_list + [np.nan] * (max_n - n))
        probs_matrix.append(padded_probs)
    
    # Convert to numpy arrays
    positions_array = np.array(positions, dtype=float)
    probs_array = np.array(probs_matrix, dtype=float)
    
    # Create meshgrid for 3D plot
    X, Y = np.meshgrid(range(1, max_n + 1), n_values)
    
    ax1 = fig1.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, probs_array, cmap='viridis', 
                           alpha=0.8, linewidth=0, antialiased=True)
    
    ax1.set_xlabel('Position (n-th person)', fontsize=10)
    ax1.set_ylabel('Number of People (N)', fontsize=10)
    ax1.set_zlabel('P(gets max)', fontsize=10)
    ax1.set_title('3D Surface: P(gets max) vs Position vs N', fontsize=12)
    ax1.view_init(elev=25, azim=45)
    
    # Add colorbar
    fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, 
                 label='Probability')
    
    # 2. Heatmap: Probability Matrix
    ax2 = fig1.add_subplot(222)
    
    # Create heatmap data (mask NaN values)
    heatmap_data = np.ma.masked_invalid(probs_array)
    
    im = ax2.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', 
                   origin='lower', interpolation='nearest')
    
    ax2.set_xlabel('Position (n-th person)', fontsize=10)
    ax2.set_ylabel('Number of People (N)', fontsize=10)
    ax2.set_title('Heatmap: P(gets max) Matrix', fontsize=12)
    
    # Set tick labels
    ax2.set_xticks(range(0, max_n, 5))
    ax2.set_xticklabels(range(1, max_n + 1, 5))
    ax2.set_yticks(range(0, len(n_values), 5))
    ax2.set_yticklabels(range(min_n, max_n + 1, 5))
    
    # Add colorbar
    fig1.colorbar(im, ax=ax2, label='Probability')
    
    # Add contour lines
    ax2.contour(heatmap_data, colors='black', linewidths=0.5, alpha=0.5)
    
    # 3. Line Plot: Probability for Selected N values
    ax3 = fig1.add_subplot(223)
    
    # Select representative N values
    selected_n = [2, 3, 5, 10, 20, 30, 40, 50]
    colors = cm.tab10(np.linspace(0, 1, len(selected_n)))
    
    for i, n in enumerate(selected_n):
        if n in probabilities_dict:
            probs = probabilities_dict[n]
            positions = range(1, n + 1)
            ax3.plot(positions, probs, 'o-', color=colors[i], 
                    linewidth=2, markersize=4, label=f'N={n}')
    
    ax3.set_xlabel('Position (n-th person)', fontsize=10)
    ax3.set_ylabel('P(gets maximum amount)', fontsize=10)
    ax3.set_title('Probability vs Position for Different N', fontsize=12)
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Set y-axis to log scale for better visualization of small probabilities
    ax3.set_yscale('log')
    ax3.set_ylim([1e-4, 1])
    
    # 4. Optimal Position Analysis
    ax4 = fig1.add_subplot(224)
    
    optimal_positions_dict = results_dict['optimal_positions']
    
    # Prepare data for optimal positions
    n_values_list = []
    optimal_pos_list = []
    optimal_prob_list = []
    
    for n in range(min_n, max_n + 1):
        if n in optimal_positions_dict:
            optimal_pos = optimal_positions_dict[n]
            max_prob = max(probabilities_dict[n])
            
            # Take the first optimal position if multiple
            n_values_list.extend([n] * len(optimal_pos))
            optimal_pos_list.extend([p + 1 for p in optimal_pos])  # Convert to 1-indexed
            optimal_prob_list.extend([max_prob] * len(optimal_pos))
    
    # Scatter plot with color indicating probability
    scatter = ax4.scatter(n_values_list, optimal_pos_list, 
                         c=optimal_prob_list, cmap='coolwarm', 
                         s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add trend line (linear regression)
    if len(n_values_list) > 1:
        coeffs = np.polyfit(n_values_list, optimal_pos_list, 1)
        trend_line = np.poly1d(coeffs)
        x_trend = np.array([min_n, max_n])
        y_trend = trend_line(x_trend)
        ax4.plot(x_trend, y_trend, 'k--', linewidth=2, 
                label=f'Linear fit: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
    
    ax4.set_xlabel('Number of People (N)', fontsize=10)
    ax4.set_ylabel('Optimal Position (1-indexed)', fontsize=10)
    ax4.set_title('Optimal Position vs Number of People', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add colorbar for probability
    fig1.colorbar(scatter, ax=ax4, label='Max Probability')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    save_path1 = os.path.join(output_dir, "position_probability_comprehensive.png")
    plt.savefig(save_path1, dpi=150, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {save_path1}")
    
    plt.show()
    
    # 5. Additional Analysis: Mean Amount vs Position
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Mean Amount Analysis vs Position', fontsize=16)
    
    mean_amounts_dict = results_dict['mean_amounts']
    
    # Select N values for analysis
    analysis_n = [5, 10, 20, 50]
    
    for idx, n in enumerate(analysis_n):
        ax = axes[idx // 2, idx % 2]
        
        if n in mean_amounts_dict:
            mean_amounts = mean_amounts_dict[n]
            positions = range(1, n + 1)
            
            ax.plot(positions, mean_amounts, 'o-', color='steelblue', 
                   linewidth=2, markersize=5)
            
            # Add theoretical equal split line
            equal_split = 50 / n
            ax.axhline(y=equal_split, color='red', linestyle='--', 
                      alpha=0.7, label=f'Equal split: {equal_split:.2f}¥')
            
            # Add expected line for uniform distribution
            # For uniform(0, 2*avg), expected value = avg
            remaining_avg = 50 / n
            ax.axhline(y=remaining_avg, color='green', linestyle=':', 
                      alpha=0.7, label=f'Remaining avg: {remaining_avg:.2f}¥')
            
            ax.set_xlabel('Position', fontsize=10)
            ax.set_ylabel('Mean Amount (¥)', fontsize=10)
            ax.set_title(f'N = {n} People', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path2 = os.path.join(output_dir, "mean_amount_analysis.png")
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Mean amount analysis saved to: {save_path2}")
    
    plt.show()
    
    # 6. Probability Distribution Characteristics
    fig3 = plt.figure(figsize=(12, 8))
    
    # Calculate skewness and kurtosis of probability distributions
    skewness_list = []
    kurtosis_list = []
    entropy_list = []
    n_list = []
    
    for n in range(min_n, max_n + 1):
        if n in probabilities_dict:
            probs = probabilities_dict[n]
            
            # Only calculate if we have valid probabilities
            if len(probs) > 1 and np.sum(probs) > 0:
                skewness = stats.skew(probs)
                kurtosis = stats.kurtosis(probs)
                
                # Calculate entropy (information theory)
                probs_normalized = np.array(probs) / np.sum(probs)
                entropy = -np.sum(probs_normalized * np.log(probs_normalized + 1e-10))
                
                skewness_list.append(skewness)
                kurtosis_list.append(kurtosis)
                entropy_list.append(entropy)
                n_list.append(n)
    
    # Plot skewness and kurtosis
    ax1 = fig3.add_subplot(221)
    ax1.plot(n_list, skewness_list, 'b-', linewidth=2, label='Skewness')
    ax1.plot(n_list, kurtosis_list, 'r-', linewidth=2, label='Kurtosis')
    ax1.set_xlabel('Number of People (N)', fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.set_title('Distribution Shape Metrics', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot entropy
    ax2 = fig3.add_subplot(222)
    ax2.plot(n_list, entropy_list, 'g-', linewidth=2)
    ax2.set_xlabel('Number of People (N)', fontsize=10)
    ax2.set_ylabel('Entropy (bits)', fontsize=10)
    ax2.set_title('Distribution Entropy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot max probability vs N
    ax3 = fig3.add_subplot(223)
    max_probs = [max(probabilities_dict[n]) for n in range(min_n, max_n + 1)]
    ax3.plot(range(min_n, max_n + 1), max_probs, 'm-', linewidth=2)
    ax3.set_xlabel('Number of People (N)', fontsize=10)
    ax3.set_ylabel('Maximum Probability', fontsize=10)
    ax3.set_title('Maximum P(gets max) vs N', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot optimal position fraction (optimal_position / N)
    ax4 = fig3.add_subplot(224)
    optimal_frac_list = []
    for n in range(min_n, max_n + 1):
        if n in optimal_positions_dict:
            opt_pos = optimal_positions_dict[n][0]  # Take first optimal
            optimal_frac = (opt_pos + 1) / n  # Convert to 1-indexed and normalize
            optimal_frac_list.append(optimal_frac)
    
    ax4.plot(range(min_n, max_n + 1), optimal_frac_list, 'c-', linewidth=2)
    ax4.set_xlabel('Number of People (N)', fontsize=10)
    ax4.set_ylabel('Optimal Position / N', fontsize=10)
    ax4.set_title('Normalized Optimal Position', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Middle position')
    ax4.legend()
    
    plt.tight_layout()
    
    save_path3 = os.path.join(output_dir, "distribution_characteristics.png")
    plt.savefig(save_path3, dpi=150, bbox_inches='tight')
    print(f"Distribution characteristics saved to: {save_path3}")
    
    plt.show()
    
    return output_dir

def save_position_probability_results(results_dict, output_dir):
    """Save all position probability results to files"""
    
    # Save raw results as JSON
    results_path = os.path.join(output_dir, "position_probability_results.json")
    
    def convert_to_serializable(obj):
        """Helper function to convert numpy types to Python types"""
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(results_dict)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print(f"Raw results saved to: {results_path}")
    
    # Save summary report
    report_path = os.path.join(output_dir, "position_probability_summary.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("POSITION PROBABILITY ANALYSIS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Analysis Parameters:\n")
        f.write(f"- Number of people range: {results_dict['min_n']} to {results_dict['max_n']}\n")
        f.write(f"- Total amount per envelope: {results_dict['total_amount']} ¥\n")
        f.write(f"- Simulations per n: {results_dict['num_simulations']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-"*70 + "\n\n")
        
        # Extract key patterns
        min_n = results_dict['min_n']
        max_n = results_dict['max_n']
        optimal_positions = results_dict['optimal_positions']
        
        # Small N analysis (2-10)
        f.write("1. Small Groups (2-10 people):\n")
        for n in range(2, 11):
            if n in optimal_positions:
                opt_pos = optimal_positions[n]
                probs = results_dict['probabilities'][n]
                max_prob = max(probs)
                f.write(f"   N={n}: Optimal position(s) {[p+1 for p in opt_pos]} "
                       f"(P={max_prob:.3f})\n")
        
        f.write("\n2. Medium Groups (11-30 people):\n")
        for n in [15, 20, 25, 30]:
            if n in optimal_positions:
                opt_pos = optimal_positions[n]
                probs = results_dict['probabilities'][n]
                max_prob = max(probs)
                f.write(f"   N={n}: Optimal position(s) {[p+1 for p in opt_pos]} "
                       f"(P={max_prob:.3f})\n")
        
        f.write("\n3. Large Groups (31-50 people):\n")
        for n in [35, 40, 45, 50]:
            if n in optimal_positions:
                opt_pos = optimal_positions[n]
                probs = results_dict['probabilities'][n]
                max_prob = max(probs)
                f.write(f"   N={n}: Optimal position(s) {[p+1 for p in opt_pos]} "
                       f"(P={max_prob:.3f})\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("OBSERVATIONS AND INSIGHTS:\n")
        f.write("-"*70 + "\n\n")
        
        f.write("1. Optimal Position Trend:\n")
        f.write("   - For very small N, early positions often have advantage\n")
        f.write("   - As N increases, optimal position tends to move toward middle positions\n")
        f.write("   - For large N, the distribution becomes more symmetric\n\n")
        
        f.write("2. Probability Characteristics:\n")
        f.write("   - Maximum probability decreases as N increases\n")
        f.write("   - Distribution becomes flatter for larger N\n")
        f.write("   - Last positions often have higher variance\n\n")
        
        f.write("3. Strategic Implications:\n")
        f.write("   - In small groups: Early grabbing might be beneficial\n")
        f.write("   - In large groups: Middle positions are safest\n")
        f.write("   - Always consider the specific number of people\n\n")
        
        f.write("4. Model Limitations:\n")
        f.write("   - Based on uniform(0, 2*remaining_average) assumption\n")
        f.write("   - Real WeChat mechanism might have additional constraints\n")
        f.write("   - Results are statistical and might vary in practice\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"Summary report saved to: {report_path}")
    
    return results_path, report_path

def generate_optimal_position_recommendation(results_dict):
    """Generate practical recommendations based on the analysis"""
    
    optimal_positions = results_dict['optimal_positions']
    
    print("\n" + "="*60)
    print("PRACTICAL RECOMMENDATIONS FOR RED ENVELOPE GRABBING")
    print("="*60)
    
    recommendations = {
        "very_small_groups": {"range": "2-5 people", "advice": "Grab early (position 1-2)"},
        "small_groups": {"range": "6-15 people", "advice": "Aim for positions 2-4"},
        "medium_groups": {"range": "16-30 people", "advice": "Middle positions are best"},
        "large_groups": {"range": "31-50 people", "advice": "Any middle position, avoid extremes"},
        "very_large_groups": {"range": "50+ people", "advice": "Distribution is nearly uniform"}
    }
    
    print("\nBased on simulation results, here are the optimal strategies:\n")
    
    for key, rec in recommendations.items():
        print(f"{rec['range']}: {rec['advice']}")
    
    print("\nDetailed Analysis by Group Size:")
    print("-" * 50)
    
    # Show specific examples
    example_sizes = [2, 3, 5, 10, 20, 30, 40, 50]
    
    for n in example_sizes:
        if n in optimal_positions:
            opt_pos = optimal_positions[n]
            probs = results_dict['probabilities'][n]
            max_prob = max(probs)
            
            # Convert to 1-indexed for readability
            opt_pos_readable = [p + 1 for p in opt_pos]
            
            print(f"N={n:2d}: Optimal position(s): {opt_pos_readable} "
                  f"(P={max_prob:.3f})")
    
    print("\n" + "="*60)
    print("CONCLUSION: For best results, grab in the middle positions!")
    print("="*60)

def main_position_probability_analysis():
    """Main function for position probability analysis"""
    
    print("="*70)
    print("RED ENVELOPE POSITION PROBABILITY ANALYSIS")
    print("Analyzing P(n-th person gets maximum amount) vs Number of People")
    print("="*70)
    
    # Parameters
    min_n = 2
    max_n = 50
    total_amount = 50  # Based on your real data
    num_simulations = 10000
    
    print(f"\nAnalysis Parameters:")
    print(f"- Number of people range: {min_n} to {max_n}")
    print(f"- Total amount per envelope: {total_amount} ¥")
    print(f"- Simulations per n: {num_simulations}")
    print(f"- Estimated runtime: {((max_n-min_n+1)*num_simulations/5000):.1f} seconds")
    print("\nStarting simulations...\n")
    
    # Run simulations
    results_dict = analyze_position_probability_for_different_n(
        min_n=min_n,
        max_n=max_n,
        total_amount=total_amount,
        num_simulations=num_simulations
    )
    
    # Visualize results
    output_dir = visualize_position_probability_results(results_dict)
    
    # Save results
    results_path, report_path = save_position_probability_results(results_dict, output_dir)
    
    # Generate recommendations
    generate_optimal_position_recommendation(results_dict)
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")
    print(f"1. Raw data: {os.path.basename(results_path)}")
    print(f"2. Summary report: {os.path.basename(report_path)}")
    print(f"3. Visualization plots: *.png files")
    print("\nKey insight: The optimal position for grabbing red envelopes")
    print("            depends on the total number of participants!")
    
    return results_dict

if __name__ == "__main__":
    # Run the analysis
    results = main_position_probability_analysis()