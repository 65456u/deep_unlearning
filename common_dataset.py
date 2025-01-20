import torch
from torch.utils.data import Dataset
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index
import os

def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs):
    question_start_token = model_configs.get('question_start_tag', "<Q>")
    question_end_token = model_configs.get('question_end_tag', "</Q>")
    answer_token = model_configs.get('answer_tag', "<A>")
    
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    
    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    
    # Change label to -100 for question tokens
    for i in range(num_question_tokens):
        label[i] = -100
    
    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)

class CommonDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_configs, max_length=512, question_key='question', answer_key='answer'):
        """
        初始化 CommonDataset。

        参数：
            data_path (str): 数据文件的路径，应该是通过 `torch.save` 序列化的文件，包含数据字典。
            tokenizer (Tokenizer): 用于将文本转换为模型可接受的输入ID。
            model_configs (dict): 包含模型相关的配置参数，如特殊标记（例如 `question_start_tag`、`question_end_tag`、`answer_tag`）。
            max_length (int, optional): 输入序列的最大长度。默认值为 512。
            question_key (str, optional): 数据集中用于访问问题的键名。默认值为 'question'。
            answer_key (str, optional): 数据集中用于访问答案的键名。默认值为 'answer'。
        """
        super(CommonDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.Dataset.from_dict(torch.load(data_path))
        self.data = add_dataset_index(self.data)
        self.qk = question_key
        self.ak = answer_key
        self.model_configs = model_configs

    def __len__(self):
        """
        返回数据集的长度，基于数据集中样本的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引 `idx` 返回对应的数据样本。

        参数：
            idx (int): 数据样本的索引。

        返回：
            tuple: 包含 `input_ids`、`labels`、`attention_mask` 和 `index` 的元组。
        """
        sample = self.data[idx]
        question = sample[self.qk]
        answers = sample[self.ak]
        indices = sample.get('index', idx)  # 如果没有 'index' 字段，使用 idx 作为索引

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(
                self.tokenizer, 
                self.max_length, 
                question, 
                answer, 
                self.model_configs
            )
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        # 将列表中的张量堆叠，并去除单维度
        input_ids = torch.stack(pad_input_ids_list).squeeze()
        labels = torch.stack(label_list).squeeze()
        attention_mask = torch.stack(pad_attention_mask_list).squeeze()

        return input_ids, labels, attention_mask, torch.tensor(indices)

    def to_csv(self, output_path: str = 'common_dataset.csv'):
        """
        将数据集导出为 CSV 文件。

        参数：
            output_path (str, optional): 输出 CSV 文件的路径。默认值为 'common_dataset.csv'。
        """
        df = self.data.to_pandas()
        df.to_csv(output_path, index=False)

def custom_data_collator(samples):
    """
    自定义数据整理函数，用于在数据加载过程中批量处理样本。

    参数：
        samples (list of tuples): 每个样本是一个包含 `input_ids`、`labels`、`attention_mask` 和 `index` 的元组。

    返回:
        tuple: 堆叠后的 `input_ids`、`labels` 和 `attention_mask` 张量。
    """
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]  # 如果需要使用索引，可以返回它们

    # 堆叠张量
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels)
    attention_mask = torch.stack(attention_mask)
    indices = torch.stack(indices)

    return input_ids, labels, attention_mask, indices
