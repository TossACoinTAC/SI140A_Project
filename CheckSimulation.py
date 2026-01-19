import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chisquare, ttest_ind
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

# ==================== 2. Define Simulation Function ====================
def simulate_one_group(total=50, n=6):
    amounts = []
    remaining = total
    for i in range(n - 1):
        avg_remaining = remaining / (n - i)
        draw = np.random.uniform(0, 2 * avg_remaining)
        amounts.append(draw)
        remaining -= draw
    amounts.append(remaining)  # last person gets the rest
    return amounts

def simulate_many_groups(num_groups=200, total=50, n=6):
    groups = []
    for _ in range(num_groups):
        groups.append(simulate_one_group(total, n))
    return groups

# ==================== 3. Generate Simulated Data ====================
np.random.seed(42)
simulated_data = simulate_many_groups(num_groups=200, total=50, n=6)
simulated_data_flat = np.array(simulated_data).flatten()
print(f"\nSimulated data: {len(simulated_data)} groups, {len(simulated_data_flat)} individual amounts")
print(f"Simulated data range: {simulated_data_flat.min():.2f} ~ {simulated_data_flat.max():.2f}")
print(f"Simulated data mean: {simulated_data_flat.mean():.2f}, std: {simulated_data_flat.std():.2f}")

# ==================== 4. Visualization ====================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Histograms
axes[0, 0].hist(real_data_flat, bins=30, alpha=0.7, color='blue', label='Real', density=True)
axes[0, 0].hist(simulated_data_flat, bins=30, alpha=0.7, color='orange', label='Simulated', density=True)
axes[0, 0].set_title('Histogram Comparison (Density)')
axes[0, 0].set_xlabel('Amount (RMB)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

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

# Scatter of group totals
real_totals = [sum(g) for g in real_data]
sim_totals = [sum(g) for g in simulated_data]
axes[1, 0].scatter(range(len(real_totals)), real_totals, alpha=0.6, label='Real', color='blue')
axes[1, 0].scatter(range(len(sim_totals)), sim_totals, alpha=0.6, label='Simulated', color='orange')
axes[1, 0].set_title('Total Amount per Group')
axes[1, 0].set_xlabel('Group Index')
axes[1, 0].set_ylabel('Total Amount (RMB)')
axes[1, 0].legend()

# QQ plot
quantiles = np.linspace(0.01, 0.99, 100)
real_quantiles = np.quantile(real_data_flat, quantiles)
sim_quantiles = np.quantile(simulated_data_flat, quantiles)
axes[1, 1].scatter(real_quantiles, sim_quantiles, alpha=0.7)
axes[1, 1].plot([real_quantiles.min(), real_quantiles.max()],
                [real_quantiles.min(), real_quantiles.max()], 'r--')
axes[1, 1].set_title('QQ Plot (Real vs Simulated)')
axes[1, 1].set_xlabel('Real Quantiles')
axes[1, 1].set_ylabel('Simulated Quantiles')

# Kernel Density Estimate
sns.kdeplot(real_data_flat, ax=axes[1, 2], label='Real', fill=True, alpha=0.5, color='blue')
sns.kdeplot(simulated_data_flat, ax=axes[1, 2], label='Simulated', fill=True, alpha=0.5, color='orange')
axes[1, 2].set_title('KDE Comparison')
axes[1, 2].set_xlabel('Amount (RMB)')
axes[1, 2].set_ylabel('Density')
axes[1, 2].legend()

plt.tight_layout()
plt.show()

# ==================== 5. Statistical Tests ====================
print("\n" + "="*50)
print("Statistical Tests for Distribution Comparison")
print("="*50)

# Kolmogorov-Smirnov test
ks_stat, ks_p = ks_2samp(real_data_flat, simulated_data_flat)
print(f"Kolmogorov-Smirnov Test:")
print(f"  KS statistic = {ks_stat:.4f}")
print(f"  p-value = {ks_p:.4f}")
if ks_p > 0.05:
    print("  → Cannot reject H0: Same distribution (p > 0.05)")
else:
    print("  → Reject H0: Different distributions (p ≤ 0.05)")

# Chi-square test
print(f"\nChi-square Goodness-of-fit Test:")
bins = np.linspace(0, 40, 15)
real_hist, bin_edges = np.histogram(real_data_flat, bins=bins)
sim_hist, _ = np.histogram(simulated_data_flat, bins=bins)

# 确保没有零频数
real_hist = np.where(real_hist == 0, 0.5, real_hist)
sim_hist = np.where(sim_hist == 0, 0.5, sim_hist)

# 计算模拟数据的概率分布
sim_probs = sim_hist / np.sum(sim_hist)

# 计算期望频数：模拟数据的概率分布 × 真实数据的总样本数
expected_freq = sim_probs * np.sum(real_hist)

# 调整期望频数以确保总和完全一致
expected_freq = expected_freq * (np.sum(real_hist) / np.sum(expected_freq))

print(f"  Real data total: {np.sum(real_hist)}")
print(f"  Expected total: {np.sum(expected_freq)}")
print(f"  Difference: {np.abs(np.sum(real_hist) - np.sum(expected_freq)):.10f}")

# 检查是否有足够的数据进行卡方检验
valid_mask = expected_freq >= 5
if np.sum(valid_mask) >= 3:
    # 使用有效区间
    real_hist_valid = real_hist[valid_mask]
    expected_freq_valid = expected_freq[valid_mask]
    
    # 再次确保总和一致
    expected_freq_valid = expected_freq_valid * (np.sum(real_hist_valid) / np.sum(expected_freq_valid))
    
    try:
        chi2_stat, chi2_p = chisquare(f_obs=real_hist_valid, f_exp=expected_freq_valid)
        print(f"  Chi2 statistic = {chi2_stat:.4f}")
        print(f"  p-value = {chi2_p:.4f}")
        print(f"  Degrees of freedom = {len(real_hist_valid) - 1}")
        print(f"  Number of bins used = {len(real_hist_valid)}")
        
        if chi2_p > 0.05:
            print("  → Cannot reject H0: Good fit (p > 0.05)")
        else:
            print("  → Reject H0: Poor fit (p ≤ 0.05)")
    except Exception as e:
        print(f"  Chi-square test failed: {e}")
        chi2_stat, chi2_p = None, None
else:
    print("  Skipped: insufficient bins with expected frequency ≥ 5")
    print(f"  Valid bins: {np.sum(valid_mask)} (need at least 3)")
    chi2_stat, chi2_p = None, None

# Two-sample t-test for means
t_stat, t_p = ttest_ind(real_data_flat, simulated_data_flat, equal_var=False)
print(f"\nWelch's t-test for means:")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value = {t_p:.4f}")
if t_p > 0.05:
    print("  → Cannot reject H0: Same mean (p > 0.05)")
else:
    print("  → Reject H0: Different means (p ≤ 0.05)")

# Additional descriptive statistics
print(f"\nDescriptive Statistics:")
real_skew = np.mean((real_data_flat - real_data_flat.mean())**3) / (real_data_flat.std()**3)
sim_skew = np.mean((simulated_data_flat - simulated_data_flat.mean())**3) / (simulated_data_flat.std()**3)
real_kurt = np.mean((real_data_flat - real_data_flat.mean())**4) / (real_data_flat.std()**4)
sim_kurt = np.mean((simulated_data_flat - simulated_data_flat.mean())**4) / (simulated_data_flat.std()**4)

print(f"  Real data skewness: {real_skew:.4f}")
print(f"  Simulated data skewness: {sim_skew:.4f}")
print(f"  Real data kurtosis: {real_kurt:.4f}")
print(f"  Simulated data kurtosis: {sim_kurt:.4f}")

# ==================== 6. Conclusion ====================
print("\n" + "="*50)
print("Conclusion")
print("="*50)

# 综合评估
test_results = {}
test_results["KS_test"] = ks_p > 0.05
test_results["T_test"] = t_p > 0.05

if chi2_p is not None:
    test_results["Chi2_test"] = chi2_p > 0.05
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
else:
    # 如果没有卡方检验结果，只考虑其他测试
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)

if total_tests > 0:
    pass_rate = passed_tests / total_tests
    
    if pass_rate >= 0.75:  # 通过75%以上的测试
        print("The simulated data (0–2×avg uniform model) is statistically consistent")
        print("with the provided WeChat Red Envelope sample data.")
        print("The model reasonably captures the underlying random mechanism.")
    elif pass_rate >= 0.5:  # 通过50%以上的测试
        print("The simulated data shows moderate similarity to the WeChat data.")
        print("The 0–2×avg uniform model captures some aspects but may need refinement.")
        print("Possible factors: rounding, minimum amounts, or correlation between draws.")
    else:
        print("The simulated data does NOT match the provided WeChat data closely.")
        print("The actual WeChat mechanism likely differs significantly from the simple")
        print("0–2×avg uniform model. Consider more complex modeling approaches.")
    
    print(f"\nTest results summary ({passed_tests}/{total_tests} tests passed):")
    for test, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test}: {status}")
else:
    print("Insufficient test results for conclusion.")

print(f"\nKey differences:")
print(f"  Real data mean: {real_data_flat.mean():.2f}, Simulated mean: {simulated_data_flat.mean():.2f}")
print(f"  Real data std: {real_data_flat.std():.2f}, Simulated std: {simulated_data_flat.std():.2f}")
print(f"  Real data min: {real_data_flat.min():.2f}, Simulated min: {simulated_data_flat.min():.2f}")
print(f"  Real data max: {real_data_flat.max():.2f}, Simulated max: {simulated_data_flat.max():.2f}")

# ==================== 7. Save Simulated Data (Optional) ====================
with open('simulated_red_envelopes.json', 'w') as f:
    json.dump(simulated_data, f, indent=2)
print("\nSimulated data saved to 'simulated_red_envelopes.json'")