import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2_contingency
import os

# 设置中文字体（保持兼容性）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 读取真实数据
DATABASE_DIR = "Database"
REAL_DATA_PATH = os.path.join(DATABASE_DIR, "Wechat_Samples.json")

def load_and_clean_real_data():
    """Load and clean real WeChat red envelope data"""
    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(f"Real data file not found: {REAL_DATA_PATH}")
    
    with open(REAL_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} raw red envelope samples")
    
    # 检查数据形状
    print("\nChecking data shape...")
    length_counts = {}
    for i, sample in enumerate(data):
        length = len(sample)
        if length not in length_counts:
            length_counts[length] = 0
        length_counts[length] += 1
    
    print("Length distribution:")
    for length, count in sorted(length_counts.items()):
        print(f"  {length} elements: {count} samples")
    
    # 只保留6个元素的数据
    cleaned_data = [sample for sample in data if len(sample) == 6]
    print(f"\nKeeping {len(cleaned_data)} samples with 6 elements")
    
    # 检查是否有无效数据（如全零）
    valid_data = []
    for sample in cleaned_data:
        # 检查是否全为零或总和为零
        if sum(sample) > 0.1 and max(sample) > 0.1:  # 避免数值误差
            valid_data.append(sample)
        else:
            print(f"Removing invalid sample (sum={sum(sample):.2f}): {sample}")
    
    print(f"Final valid data: {len(valid_data)} samples")
    
    return valid_data

def wechat_red_envelope_simulation(total_amount, num_people, num_simulations=1000):
    """
    Simulate WeChat red envelope distribution mechanism
    
    Algorithm based on known mechanism:
    - Each draw follows Uniform(0, 2 * remaining_average)
    - Where remaining_average = remaining_amount / remaining_people
    
    Parameters:
    - total_amount: total money in red envelope
    - num_people: number of people to split the money
    - num_simulations: number of simulations to run
    
    Returns:
    - simulated_data: list of simulated red envelope distributions
    """
    simulated_data = []
    
    for sim_idx in range(num_simulations):
        # Initialize
        remaining_amount = total_amount
        remaining_people = num_people
        allocations = []
        
        for person_idx in range(num_people - 1):
            # Calculate remaining average
            if remaining_people > 0:
                remaining_average = remaining_amount / remaining_people
                
                # Generate random amount: Uniform(0, 2 * remaining_average)
                # Ensure it doesn't exceed remaining_amount
                max_amount = min(2 * remaining_average, remaining_amount - 0.01)
                amount = np.random.uniform(0.01, max_amount)
                
                # Round to 2 decimal places (like real data)
                amount = round(amount, 2)
                
                allocations.append(amount)
                
                # Update remaining values
                remaining_amount -= amount
                remaining_people -= 1
            else:
                break
        
        # Last person gets all remaining amount
        last_amount = round(remaining_amount, 2)
        allocations.append(last_amount)
        
        # Ensure sum equals total amount (account for rounding errors)
        total_allocated = sum(allocations)
        if abs(total_allocated - total_amount) > 0.01:
            # Adjust last amount to match total
            allocations[-1] = round(allocations[-1] + (total_amount - total_allocated), 2)
        
        # Verify sum
        if abs(sum(allocations) - total_amount) > 0.01:
            print(f"Warning: Sum mismatch in simulation {sim_idx}: {sum(allocations)} vs {total_amount}")
        
        simulated_data.append(allocations)
        
        # Progress indicator
        if (sim_idx + 1) % 100 == 0:
            print(f"Simulated {sim_idx + 1}/{num_simulations} red envelopes")
    
    return simulated_data

def analyze_distribution(data, title_prefix="", save_path=None):
    """
    Analyze and visualize distribution of red envelope data
    
    Parameters:
    - data: list of red envelope allocations
    - title_prefix: prefix for plot titles
    - save_path: directory to save plots
    """
    # Convert to numpy array for easier manipulation
    # First ensure all samples have same length
    sample_lengths = [len(sample) for sample in data]
    if len(set(sample_lengths)) > 1:
        print(f"Warning: Inconsistent sample lengths: {set(sample_lengths)}")
        # Use the most common length
        from collections import Counter
        length_counter = Counter(sample_lengths)
        common_length = length_counter.most_common(1)[0][0]
        print(f"Using most common length: {common_length}")
        data = [sample for sample in data if len(sample) == common_length]
    
    data_array = np.array(data)
    num_samples, num_people = data_array.shape
    
    print(f"\n{title_prefix} Analysis:")
    print(f"Number of samples: {num_samples}")
    print(f"Number of people per sample: {num_people}")
    print(f"Total money per sample: {sum(data_array[0]):.2f} (constant for all)")
    
    # Statistical summary
    print("\nStatistical summary by position (mean ± std):")
    for i in range(num_people):
        position_data = data_array[:, i]
        print(f"Position {i}: {np.mean(position_data):.2f} ± {np.std(position_data):.2f} "
              f"[{np.min(position_data):.2f}, {np.max(position_data):.2f}]")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} Distribution Analysis', fontsize=16)
    
    # Plot 1: Overlay histogram of all positions
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, num_people))
    
    for i in range(num_people):
        ax1.hist(data_array[:, i], bins=30, alpha=0.5, 
                label=f'Pos {i}', color=colors[i], density=True)
    
    ax1.set_xlabel('Amount (¥)')
    ax1.set_ylabel('Density')
    ax1.set_title('All Positions Overlay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by position
    ax2 = axes[0, 1]
    box_data = [data_array[:, i] for i in range(num_people)]
    bp = ax2.boxplot(box_data, positions=range(num_people), patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_xlabel('Position Index')
    ax2.set_ylabel('Amount (¥)')
    ax2.set_title('Distribution by Position (Box Plot)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean amount by position
    ax3 = axes[0, 2]
    means = [np.mean(data_array[:, i]) for i in range(num_people)]
    stds = [np.std(data_array[:, i]) for i in range(num_people)]
    
    ax3.errorbar(range(num_people), means, yerr=stds, 
                fmt='o-', capsize=5, capthick=2)
    ax3.axhline(y=sum(data_array[0])/num_people, color='r', 
               linestyle='--', label='Equal Split')
    
    ax3.set_xlabel('Position Index')
    ax3.set_ylabel('Mean Amount (¥)')
    ax3.set_title('Mean Amount by Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot of amounts
    ax4 = axes[1, 0]
    for i in range(num_people):
        x_pos = np.random.normal(i, 0.05, size=num_samples)  # Add jitter
        ax4.scatter(x_pos, data_array[:, i], alpha=0.5, 
                   s=20, color=colors[i], label=f'Pos {i}' if i==0 else "")
    
    ax4.set_xlabel('Position Index')
    ax4.set_ylabel('Amount (¥)')
    ax4.set_title('Amount Distribution (Scatter)')
    ax4.set_xticks(range(num_people))
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Cumulative distribution by position
    ax5 = axes[1, 1]
    for i in range(num_people):
        sorted_data = np.sort(data_array[:, i])
        cum_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax5.plot(sorted_data, cum_prob, label=f'Pos {i}', color=colors[i])
    
    ax5.set_xlabel('Amount (¥)')
    ax5.set_ylabel('Cumulative Probability')
    ax5.set_title('Cumulative Distribution by Position')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Position correlation heatmap (simplified)
    ax6 = axes[1, 2]
    correlation_matrix = np.corrcoef(data_array.T)
    im = ax6.imshow(correlation_matrix, cmap='coolwarm', 
                   vmin=-1, vmax=1, aspect='auto')
    
    # Add correlation values
    for i in range(num_people):
        for j in range(num_people):
            text = ax6.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", 
                           color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")
    
    ax6.set_xlabel('Position Index')
    ax6.set_ylabel('Position Index')
    ax6.set_title('Position Correlation Matrix')
    ax6.set_xticks(range(num_people))
    ax6.set_yticks(range(num_people))
    
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # Clean filename
        clean_title = title_prefix.lower().replace(" ", "_").replace("-", "_")
        filename = os.path.join(save_path, f'{clean_title}_analysis.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    plt.show()
    
    return data_array

def compare_real_vs_simulated(real_data, simulated_data):
    """
    Compare real and simulated data using statistical tests
    
    Parameters:
    - real_data: real WeChat red envelope data
    - simulated_data: simulated red envelope data
    
    Returns:
    - test_results: dictionary containing test statistics
    """
    # Convert to numpy arrays
    real_array = np.array(real_data)
    sim_array = np.array(simulated_data)
    
    # Ensure same dimensions
    min_samples = min(len(real_array), len(sim_array))
    real_array = real_array[:min_samples]
    sim_array = sim_array[:min_samples]
    
    test_results = {}
    
    print("\n" + "="*60)
    print("COMPARISON: REAL vs SIMULATED DATA")
    print("="*60)
    
    # 1. KS Test for each position
    print("\n1. Kolmogorov-Smirnov Test by Position:")
    print("-" * 40)
    
    ks_results = []
    for i in range(real_array.shape[1]):
        ks_stat, p_value = ks_2samp(real_array[:, i], sim_array[:, i])
        ks_results.append({
            'position': i,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'same_distribution': p_value > 0.05
        })
        
        print(f"Position {i}: KS = {ks_stat:.4f}, p = {p_value:.4f} "
              f"{'✓ Same dist' if p_value > 0.05 else '✗ Different dist'}")
    
    test_results['ks_by_position'] = ks_results
    
    # 2. Overall KS test (flatten all data)
    print("\n2. Overall KS Test (all data flattened):")
    print("-" * 40)
    
    ks_stat_overall, p_value_overall = ks_2samp(real_array.flatten(), sim_array.flatten())
    test_results['ks_overall'] = {
        'ks_statistic': ks_stat_overall,
        'p_value': p_value_overall,
        'same_distribution': p_value_overall > 0.05
    }
    
    print(f"Overall: KS = {ks_stat_overall:.4f}, p = {p_value_overall:.4f} "
          f"{'✓ Same distribution' if p_value_overall > 0.05 else '✗ Different distribution'}")
    
    # 3. Compare means and variances
    print("\n3. Statistical Summary Comparison:")
    print("-" * 40)
    print(f"{'Metric':<15} {'Real':<15} {'Simulated':<15} {'Difference':<15}")
    print("-" * 60)
    
    real_mean = np.mean(real_array.flatten())
    sim_mean = np.mean(sim_array.flatten())
    real_std = np.std(real_array.flatten())
    sim_std = np.std(sim_array.flatten())
    
    print(f"{'Mean':<15} {real_mean:<15.4f} {sim_mean:<15.4f} {abs(real_mean-sim_mean):<15.4f}")
    print(f"{'Std Dev':<15} {real_std:<15.4f} {sim_std:<15.4f} {abs(real_std-sim_std):<15.4f}")
    
    test_results['summary_stats'] = {
        'real_mean': real_mean,
        'sim_mean': sim_mean,
        'real_std': real_std,
        'sim_std': sim_std
    }
    
    # 4. Visual comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Real vs Simulated Data Comparison', fontsize=16)
    
    # Plot 1: Overlay histograms
    ax1 = axes[0, 0]
    ax1.hist(real_array.flatten(), bins=30, alpha=0.7, 
             label='Real Data', density=True, color='blue')
    ax1.hist(sim_array.flatten(), bins=30, alpha=0.7, 
             label='Simulated Data', density=True, color='red')
    ax1.set_xlabel('Amount (¥)')
    ax1.set_ylabel('Density')
    ax1.set_title('Overall Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison by position
    ax2 = axes[0, 1]
    positions = list(range(real_array.shape[1]))
    box_data_real = [real_array[:, i] for i in positions]
    box_data_sim = [sim_array[:, i] for i in positions]
    
    # Real data boxplot
    bp1 = ax2.boxplot(box_data_real, positions=[p-0.2 for p in positions], 
                     widths=0.35, patch_artist=True)
    # Simulated data boxplot
    bp2 = ax2.boxplot(box_data_sim, positions=[p+0.2 for p in positions], 
                     widths=0.35, patch_artist=True)
    
    # Color the boxes
    for box in bp1['boxes']:
        box.set_facecolor('blue')
        box.set_alpha(0.7)
    for box in bp2['boxes']:
        box.set_facecolor('red')
        box.set_alpha(0.7)
    
    ax2.set_xlabel('Position Index')
    ax2.set_ylabel('Amount (¥)')
    ax2.set_title('Distribution by Position')
    ax2.set_xticks(positions)
    ax2.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Simulated'])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mean comparison by position
    ax3 = axes[1, 0]
    real_means = [np.mean(real_array[:, i]) for i in positions]
    sim_means = [np.mean(sim_array[:, i]) for i in positions]
    
    ax3.plot(positions, real_means, 'bo-', label='Real Means', linewidth=2)
    ax3.plot(positions, sim_means, 'ro-', label='Simulated Means', linewidth=2)
    ax3.axhline(y=sum(real_array[0])/len(positions), color='g', 
                linestyle='--', label='Equal Split')
    
    ax3.set_xlabel('Position Index')
    ax3.set_ylabel('Mean Amount (¥)')
    ax3.set_title('Mean Amount by Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: QQ Plot (Quantile-Quantile)
    ax4 = axes[1, 1]
    # Sort both datasets
    real_sorted = np.sort(real_array.flatten())
    sim_sorted = np.sort(sim_array.flatten())
    
    # Generate quantiles
    quantiles = np.linspace(0, 1, min(len(real_sorted), len(sim_sorted)))
    real_quantiles = np.quantile(real_sorted, quantiles)
    sim_quantiles = np.quantile(sim_sorted, quantiles)
    
    ax4.scatter(real_quantiles, sim_quantiles, alpha=0.6, s=20)
    
    # Add diagonal line (perfect match)
    min_val = min(real_quantiles.min(), sim_quantiles.min())
    max_val = max(real_quantiles.max(), sim_quantiles.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 
             'r--', alpha=0.7, label='Perfect Match')
    
    ax4.set_xlabel('Real Data Quantiles (¥)')
    ax4.set_ylabel('Simulated Data Quantiles (¥)')
    ax4.set_title('QQ Plot: Real vs Simulated')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    save_dir = os.path.join(DATABASE_DIR, "Simulation_Results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, "real_vs_simulated_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    
    plt.show()
    
    return test_results

def convert_to_serializable(obj):
    """
    Convert numpy/pandas objects to Python native types for JSON serialization
    
    Parameters:
    - obj: any Python object
    
    Returns:
    - Serializable version of the object
    """
    if isinstance(obj, (np.float32, np.float64, np.float_)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int_)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        # For custom objects
        return convert_to_serializable(obj.__dict__)
    else:
        # For other types, return as-is if JSON can handle it
        return obj

def save_simulation_results(real_data, simulated_data, test_results):
    """Save all simulation results to files"""
    save_dir = os.path.join(DATABASE_DIR, "Simulation_Results")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. Save simulated data
    sim_data_path = os.path.join(save_dir, "simulated_data.json")
    try:
        with open(sim_data_path, 'w', encoding='utf-8') as f:
            json.dump(simulated_data, f, ensure_ascii=False, indent=2)
        print(f"Simulated data saved to: {sim_data_path}")
    except Exception as e:
        print(f"Error saving simulated data: {e}")
        # Try with manual conversion
        serializable_sim = convert_to_serializable(simulated_data)
        with open(sim_data_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_sim, f, ensure_ascii=False, indent=2)
        print(f"Simulated data saved (after conversion): {sim_data_path}")
    
    # 2. Save test results
    test_results_path = os.path.join(save_dir, "test_results.json")
    
    try:
        # First try to serialize directly
        serializable_results = convert_to_serializable(test_results)
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"Test results saved to: {test_results_path}")
    except Exception as e:
        print(f"Error saving test results: {e}")
        # Create a simplified version
        simplified_results = {}
        if 'ks_by_position' in test_results:
            simplified_results['ks_by_position'] = []
            for item in test_results['ks_by_position']:
                simplified_results['ks_by_position'].append({
                    'position': int(item['position']),
                    'ks_statistic': float(item['ks_statistic']),
                    'p_value': float(item['p_value']),
                    'same_distribution': bool(item['same_distribution'])
                })
        
        if 'ks_overall' in test_results:
            simplified_results['ks_overall'] = {
                'ks_statistic': float(test_results['ks_overall']['ks_statistic']),
                'p_value': float(test_results['ks_overall']['p_value']),
                'same_distribution': bool(test_results['ks_overall']['same_distribution'])
            }
        
        if 'summary_stats' in test_results:
            simplified_results['summary_stats'] = {
                'real_mean': float(test_results['summary_stats']['real_mean']),
                'sim_mean': float(test_results['summary_stats']['sim_mean']),
                'real_std': float(test_results['summary_stats']['real_std']),
                'sim_std': float(test_results['summary_stats']['sim_std'])
            }
        
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        print(f"Test results saved (simplified): {test_results_path}")
    
    # 3. Save summary report
    report_path = os.path.join(save_dir, "simulation_summary.txt")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("WECHAT RED ENVELOPE SIMULATION SUMMARY REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Real Data Samples: {len(real_data)}\n")
            f.write(f"Simulated Data Samples: {len(simulated_data)}\n")
            if len(real_data) > 0:
                f.write(f"Total Amount per Envelope: {sum(real_data[0]):.2f} ¥\n")
                f.write(f"Number of People per Envelope: {len(real_data[0])}\n")
            f.write("\n")
            
            f.write("-"*40 + "\n")
            f.write("KOLMOGOROV-SMIRNOV TEST RESULTS\n")
            f.write("-"*40 + "\n\n")
            
            # Overall KS test
            ks_overall = test_results.get('ks_overall', {})
            f.write(f"Overall KS Test:\n")
            f.write(f"  KS Statistic: {ks_overall.get('ks_statistic', 0):.4f}\n")
            f.write(f"  P-value: {ks_overall.get('p_value', 0):.4f}\n")
            f.write(f"  Conclusion: {'Same distribution' if ks_overall.get('same_distribution', False) else 'Different distribution'}\n\n")
            
            # By position KS tests
            f.write("KS Test by Position:\n")
            ks_by_pos = test_results.get('ks_by_position', [])
            for result in ks_by_pos:
                f.write(f"  Position {result.get('position', 0)}: "
                       f"KS={result.get('ks_statistic', 0):.4f}, "
                       f"p={result.get('p_value', 0):.4f}, "
                       f"{'Same' if result.get('same_distribution', False) else 'Different'}\n")
            
            f.write("\n" + "-"*40 + "\n")
            f.write("STATISTICAL SUMMARY\n")
            f.write("-"*40 + "\n\n")
            
            summary = test_results.get('summary_stats', {})
            f.write(f"Real Data Mean: {summary.get('real_mean', 0):.4f} ¥\n")
            f.write(f"Simulated Data Mean: {summary.get('sim_mean', 0):.4f} ¥\n")
            f.write(f"Difference: {abs(summary.get('real_mean', 0)-summary.get('sim_mean', 0)):.4f} ¥\n\n")
            
            f.write(f"Real Data Std Dev: {summary.get('real_std', 0):.4f} ¥\n")
            f.write(f"Simulated Data Std Dev: {summary.get('sim_std', 0):.4f} ¥\n")
            f.write(f"Difference: {abs(summary.get('real_std', 0)-summary.get('sim_std', 0)):.4f} ¥\n\n")
            
            # Conclusion
            f.write("-"*40 + "\n")
            f.write("CONCLUSION\n")
            f.write("-"*40 + "\n\n")
            
            if ks_overall.get('p_value', 0) > 0.05:
                f.write("✓ The simulated data follows the same distribution as real WeChat data.\n")
                f.write("  The proposed mechanism (Uniform(0, 2*remaining_average)) is plausible.\n")
            else:
                f.write("✗ The simulated data does NOT follow the same distribution as real WeChat data.\n")
                f.write("  The proposed mechanism may need adjustment or the real mechanism is different.\n")
            
            f.write("\n" + "="*60 + "\n")
        
        print(f"Summary report saved to: {report_path}")
        
    except Exception as e:
        print(f"Error saving summary report: {e}")
        # Create a minimal report
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"Simulation completed at: {os.path.getctime(__file__)}\n")
                f.write(f"Real samples: {len(real_data)}\n")
                f.write(f"Simulated samples: {len(simulated_data)}\n")
            print(f"Minimal report saved to: {report_path}")
        except:
            print("Failed to save any report")

def main():
    """Main function to execute the simulation and analysis"""
    print("="*60)
    print("WECHAT RED ENVELOPE SIMULATION - STEP 5")
    print("="*60)
    
    # Load and clean real data
    try:
        real_data = load_and_clean_real_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if len(real_data) == 0:
        print("Error: No valid data found!")
        return
    
    # Get parameters from real data
    num_people = len(real_data[0])
    total_amount = sum(real_data[0])  # Assuming all envelopes have same total
    num_real_samples = len(real_data)
    
    print(f"\nReal Data Parameters:")
    print(f"- Total amount per envelope: {total_amount} ¥")
    print(f"- Number of people: {num_people}")
    print(f"- Number of real samples: {num_real_samples}")
    
    # Check if all envelopes have same total
    totals = [sum(sample) for sample in real_data]
    unique_totals = set(round(t, 2) for t in totals)
    if len(unique_totals) > 1:
        print(f"\nWarning: Envelopes have different totals: {sorted(unique_totals)}")
        # Use the most common total
        from collections import Counter
        total_counter = Counter(totals)
        common_total = total_counter.most_common(1)[0][0]
        print(f"Using most common total: {common_total} ¥")
        total_amount = common_total
    
    # Step 1: Analyze real data
    print("\n" + "="*60)
    print("ANALYZING REAL WECHAT DATA")
    print("="*60)
    
    try:
        real_array = analyze_distribution(
            real_data, 
            title_prefix="Real WeChat Data",
            save_path=os.path.join(DATABASE_DIR, "Simulation_Results")
        )
    except Exception as e:
        print(f"Error analyzing real data: {e}")
        # Try with simpler analysis
        print("\nPerforming basic analysis...")
        real_array = np.array(real_data)
        print(f"Real data shape: {real_array.shape}")
        print(f"Mean by position: {np.mean(real_array, axis=0)}")
    
    # Step 2: Simulate data using known mechanism
    print("\n" + "="*60)
    print("SIMULATING DATA USING KNOWN MECHANISM")
    print("="*60)
    print("Algorithm: Uniform(0, 2 * remaining_average)")
    print(f"Parameters: total_amount={total_amount}, num_people={num_people}")
    
    # Run simulation (same number of samples as real data)
    simulated_data = wechat_red_envelope_simulation(
        total_amount=total_amount,
        num_people=num_people,
        num_simulations=num_real_samples
    )
    
    # Analyze simulated data
    try:
        sim_array = analyze_distribution(
            simulated_data,
            title_prefix="Simulated Data",
            save_path=os.path.join(DATABASE_DIR, "Simulation_Results")
        )
    except Exception as e:
        print(f"Error analyzing simulated data: {e}")
        sim_array = np.array(simulated_data)
        print(f"Simulated data shape: {sim_array.shape}")
    
    # Step 3: Compare real vs simulated
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON: REAL vs SIMULATED")
    print("="*60)
    
    test_results = compare_real_vs_simulated(real_data, simulated_data)
    
    # Step 4: Save results
    save_simulation_results(real_data, simulated_data, test_results)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    
    # Final conclusion
    ks_overall = test_results.get('ks_overall', {})
    if ks_overall.get('same_distribution', False):
        print("\n✅ CONCLUSION: The simulated data is statistically similar to real WeChat data.")
        print("   The proposed mechanism (Uniform(0, 2*remaining_average)) is plausible.")
    else:
        print("\n❌ CONCLUSION: The simulated data is statistically different from real WeChat data.")
        print("   The real mechanism might be more complex than the proposed uniform distribution.")
        print("   Consider exploring other distributions or additional constraints.")

if __name__ == "__main__":
    main()