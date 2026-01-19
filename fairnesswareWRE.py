import numpy as np
from collections import defaultdict

class FairnessAwareRedEnvelope:
    def __init__(self):
        self.history = defaultdict(list)  # user_id -> list of received amounts
        self.total_sent = 0

    def send_red_envelope(self, total_amount, users, sender=None):
        """
        发送一个红包，考虑历史公平性
        """
        n = len(users)
        if n == 0:
            return []

        # Step 1: 计算每个用户的当前平均收益
        avg_amounts = {}
        for u in users:
            if u in self.history:
                avg_amounts[u] = sum(self.history[u]) / len(self.history[u])
            else:
                avg_amounts[u] = 0.0

        # Step 2: 计算“公平分数”：越低的人越需要补偿
        fairness_scores = {u: 1.0 / (avg_amounts[u] + 0.1) for u in users}
        total_score = sum(fairness_scores.values())
        weights = {u: fairness_scores[u] / total_score for u in users}

        # Step 3: 基于权重进行带偏置的分配
        amounts = []
        remaining = total_amount
        remaining_people = n

        for i in range(n - 1):
            u = users[i]
            # 使用权重影响期望值
            expected = total_amount * weights[u]
            # 控制波动范围
            upper_bound = min(2 * expected, remaining)
            lower_bound = 0.01
            new_amount = np.random.uniform(lower_bound, upper_bound)
            new_amount = min(new_amount, remaining - (remaining_people - 1) * 0.01)
            amounts.append(round(new_amount, 2))
            remaining -= new_amount
            remaining_people -= 1

        amounts.append(round(remaining, 2))

        # Step 4: 更新历史记录
        for i, u in enumerate(users):
            self.history[u].append(amounts[i])

        return amounts

faire = FairnessAwareRedEnvelope()
users = ['A', 'B', 'C', 'D', 'E']

# 第一次发红包
print("Round 1:", faire.send_red_envelope(50.0, users))

# 第二次发红包
print("Round 2:", faire.send_red_envelope(50.0, users))

# 第三次发红包：假设 C 很不幸，总金额很低
print("Round 3:", faire.send_red_envelope(50.0, users))