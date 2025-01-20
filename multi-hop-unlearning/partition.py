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

# 遗忘数量
unlearning_targets_num = total_unlearning_targets // 10

new_dataset = {}
# 遗忘集和问答
# unlearning_targets, single_hop_questions, single_hop_answers
new_dataset["forget_set"] = []
# 保留集和问答
# # unlearning_targets, single_hop_questions, single_hop_answers
new_dataset["retain_set"] = []

new_dataset["multi_hop_questions"] = []

# targets, single_hop_questions, single_hop_answers
new_dataset["rest_set"] = []

new_dataset["multi_hop_questions_for_rest_set"] = []


i = 1
j = 1
k = 1
for d in dataset:
    if i < unlearning_targets_num:
        for unl_targ, question, answer in zip(d["unlearning_targets"], d["single_hop_questions_for_unlearning_targets"], d["single_hop_answers_for_unlearning_targets"]):

            forget_sample = {}
            forget_sample["unlearning_target_id"] = i
            forget_sample["case_from"] = d["case_id"]
            forget_sample["unlearning_target"] = unl_targ
            forget_sample["single_hop_question"] = question
            forget_sample["single_hop_answer"] = answer
            new_dataset["forget_set"].append(forget_sample)
            i += 1
        
        for ret_targ, question, answer in zip(d["retaining_targets"], d["single_hop_questions_for_retaining_targets"], d["single_hop_answers_for_retaining_targets"]):
            
            retain_sample = {}
            retain_sample["retaining_target_id"] = j
            retain_sample["case_from"] = d["case_id"]
            retain_sample["retaining_target"] = ret_targ
            retain_sample["single_hop_question"] = question
            retain_sample["single_hop_answer"] = answer
            new_dataset["retain_set"].append(retain_sample)
            j += 1
        
        multi_hop_q_a = {}
        multi_hop_q_a["case_from"] = d["case_id"]
        multi_hop_q_a["multi_hop_questions"] = d["multi_hop_questions"]
        multi_hop_q_a["multi_hop_answer"] = d["multi_hop_answer"]
        new_dataset["multi_hop_questions"].append(multi_hop_q_a)
    
    else:

        
        # 把待遗忘的和保留的都作为rest set
        for unl_targ, question, answer in zip(d["unlearning_targets"], d["single_hop_questions_for_unlearning_targets"], d["single_hop_answers_for_unlearning_targets"]):

            rest_sample = {}
            rest_sample["rest_target_id"] = k
            rest_sample["case_from"] = d["case_id"]
            rest_sample["rest_target"] = unl_targ
            rest_sample["single_hop_question"] = question
            rest_sample["single_hop_answer"] = answer
            new_dataset["rest_set"].append(rest_sample)
            k += 1
        
        for ret_targ, question, answer in zip(d["retaining_targets"], d["single_hop_questions_for_retaining_targets"], d["single_hop_answers_for_retaining_targets"]):
            
            rest_sample = {}
            rest_sample["rest_target_id"] = k
            rest_sample["case_from"] = d["case_id"]
            rest_sample["rest_targets"] = ret_targ
            rest_sample["single_hop_question"] = question
            rest_sample["single_hop_answer"] = answer
            new_dataset["rest_set"].append(rest_sample)
            k += 1

        multi_hop_q_a = {}
        multi_hop_q_a["case_from"] = d["case_id"]
        multi_hop_q_a["multi_hop_questions"] = d["multi_hop_questions"]
        multi_hop_q_a["multi_hop_answer"] = d["multi_hop_answer"]
        new_dataset["multi_hop_questions_for_rest_set"].append(multi_hop_q_a)


with open("MQuAKE_forget10.json","w") as f:
    json.dump(new_dataset, f)

