import numpy as np

def generate_children_red_envelope(total_amount=50.0, num_people=6):
    # 每人基础金额 = total / num_people
    base = total_amount / num_people
    # 加上微小扰动，但仍控制在 ±10% 范围内
    amounts = []
    remaining = total_amount

    for i in range(num_people - 1):
        # 添加小范围浮动，确保不会太离谱
        fluctuation = np.random.uniform(-0.1 * base, 0.1 * base)
        amount = max(0.01, base + fluctuation)
        # 确保不超过剩余额度
        amount = min(amount, remaining - (num_people - 1 - i) * 0.01)
        amounts.append(round(float(amount), 2))
        remaining -= amount

    last_amount = round(float(remaining), 2)
    amounts.append(last_amount)

    return amounts

def generate_company_red_envelope(total_amount=50.0, roles=['boss', 'employee', 'employee', 'employee', 'employee', 'employee']):
    # 假设 boss 不想抢太多，只拿 1～3 元作为象征
    # 其他人按正常分布
    amounts = []
    remaining = total_amount
    role_map = {'boss': 0.01, 'employee': 0.01}  # 最低限额

    # 先处理非 boss 成员
    non_boss_count = sum(1 for r in roles if r != 'boss')
    non_boss_avg = (total_amount - 3.0) / non_boss_count  # 给 boss 预留 3 元上限

    for role in roles:
        if role == 'boss':
            boss_amount = np.random.uniform(0.01, 3.0)
            amounts.append(round(boss_amount, 2))
            remaining -= boss_amount
        else:
            # 正常员工分配
            upper_bound = min(2 * non_boss_avg, remaining)
            amount = np.random.uniform(0.01, upper_bound)
            amount = min(amount, remaining - (non_boss_count - len(amounts)) * 0.01)
            amounts.append(round(amount, 2))
            remaining -= amount

    # 最后一人补足
    if len(amounts) < len(roles):
        amounts.append(round(remaining, 2))

    return amounts

def generate_family_red_envelope(total_amount=88.0, weights=None, members=None):
    if weights is None:
        # 默认权重：父母(高)、配偶/子女(中)、远亲(低)
        weights = [3.0, 3.0, 2.0, 2.0, 1.0, 0.5]
    if members is None:
        members = ['父亲', '母亲', '儿子', '女儿', '叔叔', '表哥']

    num_people = len(weights)
    normalized_weights = [w / sum(weights) for w in weights]

    amounts = []
    remaining = total_amount

    for i in range(num_people - 1):
        weight = normalized_weights[i]
        expected = total_amount * weight
        fluctuation = np.random.uniform(-0.2 * expected, 0.2 * expected)
        amount = max(0.01, expected + fluctuation)
        amount = min(amount, remaining - (num_people - 1 - i) * 0.01)
        amounts.append(round(float(amount), 2))
        remaining -= amount

    amounts.append(round(float(remaining), 2))

    return dict(zip(members, amounts))

if __name__ == "__main__":
    amounts = generate_children_red_envelope(30.0, 6)
    print("🎈 幼儿园小朋友抢红包示例 (总额30元，6个小朋友)")
    print("分配结果:", amounts)
    print("每人金额:", [f"{amount}元" for amount in amounts])
    print("最大金额:", max(amounts), "元")
    print("最小金额:", min(amounts), "元")
    print("金额差值:", max(amounts) - min(amounts), "元")

    roles = ['boss', 'employee', 'employee', 'employee', 'employee', 'employee']
    amounts = generate_company_red_envelope(100.0, roles)
    print("\n🏢 公司团队抢红包示例 (总额100元，1位老板+5位员工)")
    print("分配结果:", amounts)
    print("老板获得:", amounts[0], "元")
    print("员工获得:", [f"{amount}元" for amount in amounts[1:]])
    print("员工平均:", round(sum(amounts[1:]) / 5, 2), "元")

    weights = [3.0, 3.0, 2.0, 2.0, 1.0, 0.5]  # 父母>子女>叔叔>表哥
    members = ['父亲', '母亲', '儿子', '女儿', '叔叔', '表哥']
    result = generate_family_red_envelope(88.0, weights, members)

    print("\n👨‍👩‍👧‍👦 家庭春节红包示例 (总额88元，6位亲戚)")
    print("分配结果:")
    for member, amount in result.items():
        print(f"  {member}: {amount}元")

    print("\n金额排序(从高到低):")
    sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    for member, amount in sorted_result:
        print(f"  {member}: {amount}元")