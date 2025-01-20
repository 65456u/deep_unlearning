import json
import random

with open('MQuAKE_for_unlearning_full.json', 'r') as f:
    dataset = json.load(f)

#打乱数据集
random.shuffle(dataset)

# 统计number_of_unlearning_targets总数
total_unlearning_targets = 0
for d in dataset:
    total_unlearning_targets += d["number_of_unlearning_targets"]

print(total_unlearning_targets)