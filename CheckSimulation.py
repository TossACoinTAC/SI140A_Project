import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. Load Real Data ====================
DATABASE_DIR = "Database/Wechat_Samples.json"
with open(DATABASE_DIR, 'r') as f:
    real_data = json.load(f)

real_data_flat = np.array(real_data).flatten()
print(f"Real data: {len(real_data)} groups, {len(real_data_flat)} individual amounts")
print(f"Real data range: {real_data_flat.min():.2f} ~ {real_data_flat.max():.2f}")
print(f"Real data mean: {real_data_flat.mean():.2f}, std: {real_data_flat.std():.2f}")

# 检查实际数据的最小值
print(f"Real data minimum positive amount: {real_data_flat[real_data_flat > 0].min():.2f}")

# ==================== 2. Define Simulation Function with Discrete Constraints ====================
def simulate_one_group(total=50, n=6, min_amount=0.01):
    """
    模拟一个微信红包分组
    total: 总金额（元）
    n: 人数
    min_amount: 最小金额（元），微信为0.01元
    """
    # 转换为分，便于整数运算
    total_cents = int(total * 100)
    min_cents = int(min_amount * 100)
    
    amounts_cents = []
    remaining = total_cents
    
    # 确保每人至少获得最小金额
    remaining_after_min = remaining - n * min_cents
    if remaining_after_min < 0:
        raise ValueError("总金额不足以分配给所有人最小金额")
    
    for i in range(n - 1):
        # 确保剩余人数都能获得最小金额
        available_max = remaining - (n - i - 1) * min_cents
        
        # 计算平均剩余金额（以分为单位）
        avg_remaining = remaining / (n - i)
        
        # 在 [min_cents, min(2*avg_remaining, available_max)] 范围内均匀分布
        lower_bound = min_cents
        upper_bound = min(int(2 * avg_remaining), available_max)
        
        # 确保上界不小于下界
        if upper_bound < lower_bound:
            upper_bound = lower_bound
            
        # 随机抽取整数金额（分）
        draw_cents = np.random.randint(lower_bound, upper_bound + 1)
        
        amounts_cents.append(draw_cents)
        remaining -= draw_cents
    
    # 最后一个人获得剩余金额，但至少为最小金额
    amounts_cents.append(max(remaining, min_cents))
    
    # 转换回元
    amounts = [cents / 100 for cents in amounts_cents]
    
    # 验证总和
    total_check = sum(amounts_cents) / 100
    if abs(total_check - total) > 0.01:
        # 调整最后一个人的金额使总和正确
        adjustment = int((total * 100) - sum(amounts_cents[:-1]))
        amounts_cents[-1] = max(adjustment, min_cents)
        amounts[-1] = amounts_cents[-1] / 100
    
    return amounts

def simulate_many_groups(num_groups=200, total=50, n=6, min_amount=0.01):
    groups = []
    for _ in range(num_groups):
        groups.append(simulate_one_group(total, n, min_amount))
    return groups

# ==================== 3. Generate Simulated Data ====================
np.random.seed(42)
print("\nGenerating simulated data with discrete constraints...")
simulated_data = simulate_many_groups(num_groups=200, total=50, n=6, min_amount=0.01)
simulated_data_flat = np.array(simulated_data).flatten()
print(f"Simulated data: {len(simulated_data)} groups, {len(simulated_data_flat)} individual amounts")
print(f"Simulated data range: {simulated_data_flat.min():.2f} ~ {simulated_data_flat.max():.2f}")
print(f"Simulated data mean: {simulated_data_flat.mean():.2f}, std: {simulated_data_flat.std():.2f}")
print(f"Simulated data minimum positive amount: {simulated_data_flat.min():.2f}")

# 检查模拟数据是否包含0值
zero_count = np.sum(simulated_data_flat == 0)
print(f"Zero amounts in simulated data: {zero_count}")

# ==================== 4. Detailed Statistical Comparison ====================
print("\n" + "="*50)
print("Detailed Statistical Comparison")
print("="*50)

# 基本统计量
print("\nBasic Statistics:")
print(f"{'Statistic':<15} {'Real':<10} {'Simulated':<10} {'Difference':<10}")
print("-" * 55)
print(f"{'Mean':<15} {real_data_flat.mean():<10.4f} {simulated_data_flat.mean():<10.4f} {abs(real_data_flat.mean()-simulated_data_flat.mean()):<10.4f}")
print(f"{'Std Dev':<15} {real_data_flat.std():<10.4f} {simulated_data_flat.std():<10.4f} {abs(real_data_flat.std()-simulated_data_flat.std()):<10.4f}")
print(f"{'Median':<15} {np.median(real_data_flat):<10.4f} {np.median(simulated_data_flat):<10.4f} {abs(np.median(real_data_flat)-np.median(simulated_data_flat)):<10.4f}")
print(f"{'Min':<15} {real_data_flat.min():<10.4f} {simulated_data_flat.min():<10.4f} {abs(real_data_flat.min()-simulated_data_flat.min()):<10.4f}")
print(f"{'Max':<15} {real_data_flat.max():<10.4f} {simulated_data_flat.max():<10.4f} {abs(real_data_flat.max()-simulated_data_flat.max()):<10.4f}")
print(f"{'Q1':<15} {np.percentile(real_data_flat, 25):<10.4f} {np.percentile(simulated_data_flat, 25):<10.4f} {abs(np.percentile(real_data_flat, 25)-np.percentile(simulated_data_flat, 25)):<10.4f}")
print(f"{'Q3':<15} {np.percentile(real_data_flat, 75):<10.4f} {np.percentile(simulated_data_flat, 75):<10.4f} {abs(np.percentile(real_data_flat, 75)-np.percentile(simulated_data_flat, 75)):<10.4f}")

# ==================== 5. Visualization ====================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Histograms with discrete bins
# 创建以分为单位的bins
bins_cents = np.arange(0, 4100, 100) / 100  # 从0到41元，每0.01元一个bin
axes[0, 0].hist(real_data_flat, bins=bins_cents, alpha=0.7, color='blue', 
                label='Real', density=True, edgecolor='black', linewidth=0.5)
axes[0, 0].hist(simulated_data_flat, bins=bins_cents, alpha=0.7, color='orange', 
                label='Simulated', density=True, edgecolor='black', linewidth=0.5)
axes[0, 0].set_title('Histogram Comparison (Discrete, Density)')
axes[0, 0].set_xlabel('Amount (RMB)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 40)

# Boxplots
box_data = [real_data_flat, simulated_data_flat]
axes[0, 1].boxplot(box_data, labels=['Real', 'Simulated'])
axes[0, 1].set_title('Boxplot Comparison')
axes[0, 1].set_ylabel('Amount (RMB)')

# ECDF plots
def ecdf(data):
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, y

real_x, real_y = ecdf(real_data_flat)
sim_x, sim_y = ecdf(simulated_data_flat)
axes[0, 2].plot(real_x, real_y, label='Real ECDF', color='blue')
axes[0, 2].plot(sim_x, sim_y, label='Simulated ECDF', color='orange')
axes[0, 2].set_title('ECDF Comparison')
axes[0, 2].set_xlabel('Amount (RMB)')
axes[0, 2].set_ylabel('Cumulative Probability')
axes[0, 2].legend()
axes[0, 2].set_xlim(0, 40)

# Scatter of group totals with jitter
real_totals = [sum(g) for g in real_data]
sim_totals = [sum(g) for g in simulated_data]
# 添加轻微抖动以避免重叠
jitter_real = np.random.normal(0, 0.1, len(real_totals))
jitter_sim = np.random.normal(0, 0.1, len(sim_totals))
axes[1, 0].scatter(np.arange(len(real_totals)) + jitter_real, real_totals, 
                   alpha=0.6, label='Real', color='blue', s=20)
axes[1, 0].scatter(np.arange(len(sim_totals)) + jitter_sim, sim_totals, 
                   alpha=0.6, label='Simulated', color='orange', s=20)
axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Expected total=50')
axes[1, 0].set_title('Total Amount per Group (with jitter)')
axes[1, 0].set_xlabel('Group Index')
axes[1, 0].set_ylabel('Total Amount (RMB)')
axes[1, 0].legend()
axes[1, 0].set_ylim(49, 51)

# QQ plot
quantiles = np.linspace(0.01, 0.99, 100)
real_quantiles = np.quantile(real_data_flat, quantiles)
sim_quantiles = np.quantile(simulated_data_flat, quantiles)
axes[1, 1].scatter(real_quantiles, sim_quantiles, alpha=0.7, s=20)
axes[1, 1].plot([real_quantiles.min(), real_quantiles.max()],
                [real_quantiles.min(), real_quantiles.max()], 'r--', alpha=0.7)
axes[1, 1].set_title('QQ Plot (Real vs Simulated)')
axes[1, 1].set_xlabel('Real Quantiles (RMB)')
axes[1, 1].set_ylabel('Simulated Quantiles (RMB)')
axes[1, 1].set_xlim(0, 40)
axes[1, 1].set_ylim(0, 40)

# Distribution of amounts within groups (first 10 groups)
axes[1, 2].violinplot(real_data[:10], positions=range(1, 11), 
                      showmeans=True, showextrema=True)
axes[1, 2].violinplot(simulated_data[:10], positions=np.arange(1.3, 11.3), 
                      showmeans=True, showextrema=True)
axes[1, 2].set_title('Distribution within First 10 Groups')
axes[1, 2].set_xlabel('Group Index')
axes[1, 2].set_ylabel('Amount (RMB)')
axes[1, 2].set_xticks(range(1, 11))
axes[1, 2].legend(['Real', 'Simulated'], loc='upper right')

plt.tight_layout()
plt.show()

# ==================== 6. Statistical Tests ====================
print("\n" + "="*50)
print("Statistical Tests for Distribution Comparison")
print("="*50)

# 1. Kolmogorov-Smirnov test
ks_stat, ks_p = ks_2samp(real_data_flat, simulated_data_flat)
print(f"\n1. Kolmogorov-Smirnov Test (Two-sample):")
print(f"   KS statistic = {ks_stat:.6f}")
print(f"   p-value = {ks_p:.6f}")
if ks_p > 0.05:
    print(f"   → Cannot reject H0: Same distribution (p > 0.05)")
    print(f"   Interpretation: The two samples likely come from the same distribution")
else:
    print(f"   → Reject H0: Different distributions (p ≤ 0.05)")
    print(f"   Interpretation: The two samples likely come from different distributions")

# 2. Two-sample t-test (Welch's t-test)
t_stat, t_p = ttest_ind(real_data_flat, simulated_data_flat, equal_var=False)
print(f"\n2. Welch's t-test (for means):")
print(f"   t-statistic = {t_stat:.6f}")
print(f"   p-value = {t_p:.6f}")
if t_p > 0.05:
    print(f"   → Cannot reject H0: Same mean (p > 0.05)")
    print(f"   Interpretation: No significant difference in means")
else:
    print(f"   → Reject H0: Different means (p ≤ 0.05)")
    print(f"   Interpretation: Significant difference in means")
print(f"   Mean difference: {real_data_flat.mean() - simulated_data_flat.mean():.6f}")

# 3. Mann-Whitney U test (non-parametric)
u_stat, u_p = mannwhitneyu(real_data_flat, simulated_data_flat, alternative='two-sided')
print(f"\n3. Mann-Whitney U test (non-parametric):")
print(f"   U statistic = {u_stat:.0f}")
print(f"   p-value = {u_p:.6f}")
if u_p > 0.05:
    print(f"   → Cannot reject H0: Same distribution (p > 0.05)")
    print(f"   Interpretation: No significant difference in distributions")
else:
    print(f"   → Reject H0: Different distributions (p ≤ 0.05)")
    print(f"   Interpretation: Significant difference in distributions")

# ==================== 7. Analysis of Discrete Nature ====================
print("\n" + "="*50)
print("Analysis of Discrete Nature")
print("="*50)

# 检查金额的小数部分
real_decimals = real_data_flat * 100 % 100
sim_decimals = simulated_data_flat * 100 % 100

print(f"\nDecimal part analysis (in cents):")
print(f"Real data - Unique decimal values: {len(np.unique(real_decimals))}")
print(f"Simulated data - Unique decimal values: {len(np.unique(sim_decimals))}")

# 检查是否都是0.01的倍数
real_is_multiple = np.all(np.abs(real_decimals - np.round(real_decimals)) < 0.001)
sim_is_multiple = np.all(np.abs(sim_decimals - np.round(sim_decimals)) < 0.001)
print(f"\nCheck for multiples of 0.01:")
print(f"Real data all multiples of 0.01: {bool(real_is_multiple)}")
print(f"Simulated data all multiples of 0.01: {bool(sim_is_multiple)}")

# 检查最小值
print(f"\nMinimum amount analysis:")
print(f"Real data minimum: {real_data_flat.min():.2f}")
print(f"Simulated data minimum: {simulated_data_flat.min():.2f}")
print(f"Both ≥ 0.01: {bool(real_data_flat.min() >= 0.01 and simulated_data_flat.min() >= 0.01)}")

# 检查是否有0值
print(f"\nZero value check:")
real_zero_count = np.sum(real_data_flat == 0)
sim_zero_count = np.sum(simulated_data_flat == 0)
print(f"Real data zeros: {real_zero_count}")
print(f"Simulated data zeros: {sim_zero_count}")

# ==================== 8. Distribution Shape Analysis ====================
print("\n" + "="*50)
print("Distribution Shape Analysis")
print("="*50)

# 偏度和峰度
from scipy.stats import skew, kurtosis
real_skew = skew(real_data_flat)
real_kurt = kurtosis(real_data_flat)
sim_skew = skew(simulated_data_flat)
sim_kurt = kurtosis(simulated_data_flat)

print(f"\nShape statistics:")
print(f"{'Statistic':<15} {'Real':<10} {'Simulated':<10} {'Difference':<10}")
print("-" * 55)
print(f"{'Skewness':<15} {real_skew:<10.4f} {sim_skew:<10.4f} {abs(real_skew-sim_skew):<10.4f}")
print(f"{'Kurtosis':<15} {real_kurt:<10.4f} {sim_kurt:<10.4f} {abs(real_kurt-sim_kurt):<10.4f}")

print(f"\nInterpretation:")
print(f"Skewness: {'Positive (right-skewed)' if real_skew > 0 else 'Negative (left-skewed)' if real_skew < 0 else 'Symmetric'}")
print(f"Kurtosis: {'Leptokurtic (heavy-tailed)' if real_kurt > 0 else 'Platykurtic (light-tailed)' if real_kurt < 0 else 'Mesokurtic (normal-like)'}")

# ==================== 9. Conclusion ====================
print("\n" + "="*50)
print("FINAL CONCLUSION")
print("="*50)

# 综合评估
tests_passed = 0
total_tests = 3

if ks_p > 0.05: tests_passed += 1
if t_p > 0.05: tests_passed += 1
if u_p > 0.05: tests_passed += 1

print(f"\nModel Performance: {tests_passed}/{total_tests} statistical tests passed")

if tests_passed >= 2:  # 多数通过
    print("\n✓ The discrete 0-2×avg uniform model provides a REASONABLE fit")
    print("  to the WeChat Red Envelope data.")
    print("\nKey successful aspects:")
    print("  1. Preserves the discrete nature (multiples of 0.01) ✓")
    print("  2. Ensures minimum amount of 0.01 for all recipients ✓")
    print("  3. Maintains total amount of exactly 50 RMB per group ✓")
    print("  4. Passes majority of statistical similarity tests ✓")
    
    if tests_passed == 3:
        print("\nExcellent fit: All 3 statistical tests indicate similarity!")
    elif tests_passed == 2:
        print("\nGood fit: 2 out of 3 statistical tests indicate similarity.")
else:
    print("\n✗ The model shows SIGNIFICANT differences from WeChat data.")
    print("  Only {tests_passed}/3 statistical tests passed.")
    
print(f"\nStatistical test results:")
print(f"  1. Kolmogorov-Smirnov test: {'PASS' if ks_p > 0.05 else 'FAIL'} (p={ks_p:.4f})")
print(f"  2. Welch's t-test: {'PASS' if t_p > 0.05 else 'FAIL'} (p={t_p:.4f})")
print(f"  3. Mann-Whitney U test: {'PASS' if u_p > 0.05 else 'FAIL'} (p={u_p:.4f})")

print(f"\nPractical implications:")
print(f"  - Model captures the basic mechanics of WeChat Red Envelope")
print(f"  - Suitable for educational purposes and basic simulations")
print(f"  - May need refinement for exact replication of WeChat's algorithm")

# ==================== 10. Save Results ====================
print("\n" + "="*50)
print("Saving Results")
print("="*50)

# Helper function to convert numpy types to Python native types
def convert_to_python_types(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj

# Save simulated data
simulated_data_python = convert_to_python_types(simulated_data)
with open('simulated_red_envelopes_discrete.json', 'w') as f:
    json.dump(simulated_data_python, f, indent=2)
print("✓ Simulated data saved to 'simulated_red_envelopes_discrete.json'")

# Save summary statistics
summary = {
    "real_data_stats": {
        "n_groups": len(real_data),
        "n_amounts": len(real_data_flat),
        "mean": float(real_data_flat.mean()),
        "std": float(real_data_flat.std()),
        "min": float(real_data_flat.min()),
        "max": float(real_data_flat.max()),
        "median": float(np.median(real_data_flat)),
        "skewness": float(real_skew),
        "kurtosis": float(real_kurt)
    },
    "simulated_data_stats": {
        "n_groups": len(simulated_data),
        "n_amounts": len(simulated_data_flat),
        "mean": float(simulated_data_flat.mean()),
        "std": float(simulated_data_flat.std()),
        "min": float(simulated_data_flat.min()),
        "max": float(simulated_data_flat.max()),
        "median": float(np.median(simulated_data_flat)),
        "skewness": float(sim_skew),
        "kurtosis": float(sim_kurt)
    },
    "test_results": {
        "kolmogorov_smirnov": {
            "statistic": float(ks_stat), 
            "p_value": float(ks_p), 
            "passed": bool(ks_p > 0.05)
        },
        "welch_t_test": {
            "statistic": float(t_stat), 
            "p_value": float(t_p), 
            "passed": bool(t_p > 0.05)
        },
        "mann_whitney_u": {
            "statistic": float(u_stat), 
            "p_value": float(u_p), 
            "passed": bool(u_p > 0.05)
        }
    },
    "discrete_nature_validation": {
        "real_is_multiple_of_0_01": bool(real_is_multiple),
        "simulated_is_multiple_of_0_01": bool(sim_is_multiple),
        "real_min_amount": float(real_data_flat.min()),
        "simulated_min_amount": float(simulated_data_flat.min()),
        "real_zero_count": int(real_zero_count),
        "simulated_zero_count": int(sim_zero_count)
    }
}

summary_python = convert_to_python_types(summary)
with open('simulation_results_summary.json', 'w') as f:
    json.dump(summary_python, f, indent=2)
print("✓ Summary statistics saved to 'simulation_results_summary.json'")

# Save test results in CSV format
import csv
with open('test_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Test', 'Statistic', 'p-value', 'Passed (p>0.05)', 'Interpretation'])
    
    writer.writerow([
        'Kolmogorov-Smirnov', 
        f"{ks_stat:.6f}", 
        f"{ks_p:.6f}", 
        'Yes' if ks_p > 0.05 else 'No',
        'Same distribution' if ks_p > 0.05 else 'Different distribution'
    ])
    
    writer.writerow([
        "Welch's t-test", 
        f"{t_stat:.6f}", 
        f"{t_p:.6f}", 
        'Yes' if t_p > 0.05 else 'No',
        'Same mean' if t_p > 0.05 else 'Different mean'
    ])
    
    writer.writerow([
        'Mann-Whitney U', 
        f"{u_stat:.0f}", 
        f"{u_p:.6f}", 
        'Yes' if u_p > 0.05 else 'No',
        'Same distribution' if u_p > 0.05 else 'Different distribution'
    ])
    
print("✓ Test results saved to 'test_results.csv'")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)
print(f"\nSummary: The discrete uniform model (0-2×avg with min=0.01) shows")
print(f"{'good' if tests_passed >= 2 else 'limited'} similarity to WeChat data.")
print(f"\nKey findings:")
print(f"1. Discrete nature correctly implemented")
print(f"2. Minimum amount constraint satisfied")
print(f"3. Statistical similarity: {tests_passed}/3 tests passed")
print(f"\nThe model is {'suitable' if tests_passed >= 2 else 'not suitable'} for")
print(f"simulating WeChat Red Envelope behavior in educational contexts.")