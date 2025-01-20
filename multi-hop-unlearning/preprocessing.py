import json


with open('datasets/MQuAKE-CF-3k-v2.json', 'r') as f:
    dataset = json.load(f)

new_dataset = []

for d in dataset:

    # 编号
    facts = {}
    facts["case_id"] = d["case_id"]

    # 遗忘目标
    facts["unlearning_targets"] = []
    # 保留目标
    facts["retaining_targets"] = []

    # 单跳遗忘目标问题和答案
    facts["single_hop_questions_for_unlearning_targets"] = []
    facts["single_hop_answers_for_unlearning_targets"] = []
    # 单跳保留目标问题和答案
    facts["single_hop_questions_for_retaining_targets"] = []
    facts["single_hop_answers_for_retaining_targets"] = []    

    temp_target_true = []
    for targ in d["requested_rewrite"]:
        temp_target_true.append(targ["target_true"]["str"])
    for r in d["single_hops"]:
        if r["answer"] in temp_target_true:
            facts["unlearning_targets"].append(r["cloze"] + " " + r["answer"])
            facts["single_hop_questions_for_unlearning_targets"].append(r["question"])
            facts["single_hop_answers_for_unlearning_targets"].append(r["answer"])
        else:
            facts["retaining_targets"].append(r["cloze"] + " " + r["answer"])
            facts["single_hop_questions_for_retaining_targets"].append(r["question"])
            facts["single_hop_answers_for_retaining_targets"].append(r["answer"])

    # 多跳问题和答案
    facts["multi_hop_questions"] = []
    for ques in d["questions"]:
        facts["multi_hop_questions"].append(ques)

    facts["multi_hop_answer"] = d["answer"]

    # 遗忘目标数
    facts["number_of_unlearning_targets"] = len(facts["unlearning_targets"])
    # 保留目标数
    facts["number_of_retaining_targets"] = len(facts["retaining_targets"])
    # 跳数
    facts["number_of_hops"] = len(facts["single_hop_questions_for_unlearning_targets"]) + len(facts["single_hop_questions_for_retaining_targets"])

    new_dataset.append(facts)

with open("MQuAKE_for_unlearning.json","w") as f:
    json.dump(new_dataset, f)