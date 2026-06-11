import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# 加载数据
TS = [71.96, 66, 62.4, 70.81, 75.48]
faithfulness = [73.3, 53.3, 30, 70, 91.7]
correctness = [40, 36.7, 10, 46.7, 83.3]
BS = [73.23, 64.56, 60.98, 63.95]
Avg = [75.77, 68.36, 62.98, 70.74, 80.6]

# 斯皮尔曼相关
corr_acc_s, p_acc_s = stats.spearmanr(Avg, correctness)
corr_faith_s, p_faith_s = stats.spearmanr(TS, faithfulness)

print(f"斯皮尔曼-隐喻vs正确率：r_s = {corr_acc_s:.4f}, p = {p_acc_s:.4f}")
print(f"斯皮尔曼-隐喻vs忠实度：r_s = {corr_faith_s:.4f}, p = {p_faith_s:.4f}")

# 显著性判断
def judge_significance(p_val):
    if p_val < 0.001:
        return "*** (极显著)"
    elif p_val < 0.01:
        return "** (高度显著)"
    elif p_val < 0.05:
        return "* (显著)"
    else:
        return "ns (不显著)"

print(f"正确率相关性显著性：{judge_significance(p_acc_s)}")
print(f"忠实度相关性显著性：{judge_significance(p_faith_s)}")