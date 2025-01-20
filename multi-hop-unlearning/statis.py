import json

with open('MQuAKE_for_unlearning.json', 'r') as f:
    dataset = json.load(f)

statistics = {}

for item in dataset:
    number_of_hops = item["number_of_hops"]
    number_of_unlearning_targets = item["number_of_unlearning_targets"]
    
    key = (number_of_hops, number_of_unlearning_targets)
    
    if key in statistics:
        statistics[key] += 1
    else:
        statistics[key] = 1

result = [{"number_of_hops": k[0], "number_of_unlearning_targets": k[1], "count": v} for k, v in statistics.items()]

print(result)

'''
[{'number_of_hops': 2, 'number_of_unlearning_targets': 1, 'count': 1027}, 
{'number_of_hops': 2, 'number_of_unlearning_targets': 2, 'count': 108}, 
{'number_of_hops': 3, 'number_of_unlearning_targets': 2, 'count': 253}, 
{'number_of_hops': 3, 'number_of_unlearning_targets': 1, 'count': 871}, 
{'number_of_hops': 3, 'number_of_unlearning_targets': 3, 'count': 12}, 
{'number_of_hops': 4, 'number_of_unlearning_targets': 1, 'count': 476},
{'number_of_hops': 4, 'number_of_unlearning_targets': 2, 'count': 229}, 
{'number_of_hops': 4, 'number_of_unlearning_targets': 3, 'count': 24}]
'''