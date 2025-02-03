## collect_python_codes copy.py

```python
import os

def collect_python_files(directory):
    """
    遍历指定目录及其子目录，收集所有以 .py 结尾的Python文件路径。
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 忽略隐藏文件夹
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    return python_files

def write_to_markdown(python_files, output_file):
    """
    将收集到的Python文件内容写入到一个Markdown文件中，
    每个文件的内容都包含在一个代码块中，并附有文件名作为标题。
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for file in python_files:
            # 写入文件名作为二级标题
            md_file.write(f'## {os.path.relpath(file)}\n\n')
            md_file.write('```python\n')
            with open(file, 'r', encoding='utf-8') as py_file:
                code = py_file.read()
                md_file.write(code)
            md_file.write('\n```\n\n')

if __name__ == "__main__":
    current_directory = os.getcwd()
    python_files = collect_python_files(current_directory)
    output_markdown = 'python_code_collection.md'
    write_to_markdown(python_files, output_markdown)
    print(f"所有Python代码已被收集到 {output_markdown}")

```

## forget.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
from datasets import Dataset
import os
import gc
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from data_module import FamilyForgetDataset, custom_data_collator
from unlearn_trainer import CustomFamilyTrainerForgetting
from utils import get_model_identifiers_from_yaml
from common_dataset import CommonDataset

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    
    cfg.data_path = 'data/mhu_01_forget_dict.pt'
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["model_id"]
    # if cfg.model_path is None:
    #     cfg.model_path = model_cfg["ft_model_path"]
    cfg.model_path = 'meta-llama/Llama-3.1-8B'

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 使用 CommonDataset 代替 FamilyForgetDataset
    torch_format_dataset = CommonDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_configs=model_cfg,
        max_length=500,
        question_key='single_hop_question',
        answer_key='single_hop_answer'
    )

    torch_format_dataset.to_csv()

    # 设置学习率和训练周期
    if cfg.forget_loss == "ga":
        lr = float(model_cfg["ga_lr"])
        num_epochs = model_cfg["ga_num_epochs"]
    elif cfg.forget_loss == "npo":
        lr = float(model_cfg["npo_lr"])
        num_epochs = model_cfg["npo_num_epochs"]

    # batch_size = cfg.batch_size
    batch_size = 2
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)
    max_steps = int(num_epochs * len(torch_format_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    print(f"max_steps: {max_steps}")
    print(f"steps_per_epoch: {steps_per_epoch}")
    max_steps = 1120
    print(f"max_steps: {max_steps}")
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        # max_steps=max_steps,
        num_train_epochs=32,
        learning_rate=lr,
        bf16=True,
        bf16_full_eval=True,
        # logging_steps=max(1, max_steps // 20),
        logging_steps=1,
        logging_dir=f'{cfg.save_dir}/logs',
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_strategy="no",
        ddp_find_unused_parameters=False,
        deepspeed='config/ds_config.json',
        weight_decay=cfg.weight_decay,
        eval_steps=1,
        evaluation_strategy="steps",
        seed=cfg.seed,
    )
    print(f"Training arguments: {training_args}")

    # 判断本地是否存在 checkpoint
    import re
    path_found = False
    # for file in os.listdir(cfg.model_path):
    #     if re.search(r"pytorch.*\.bin", file):
    #         path_found = True
    #         break
        
    #     if re.search(r"model-.*\.safetensors", file):
    #         path_found = True
    #         break

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from local checkpoint")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            use_flash_attention_2=False,
            torch_dtype=torch.float16,
            token=os.environ['HF_TOKEN'],
            trust_remote_code=True
        )
    else:
        print("Local checkpoint not found. Loading from remote.")
        # 从远程加载模型
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=model_cfg.get("config", None),  # 如果需要，可以从配置中获取额外参数
            use_flash_attention_2=False,
            torch_dtype=torch.float16,
            token=os.environ.get('HF_TOKEN'),  # 确保环境变量中存在 HF_TOKEN
            trust_remote_code=True
        )
        # 如果需要，可以将远程模型保存到本地
        # model.save_pretrained(cfg.model_path)
        # tokenizer.save_pretrained(cfg.model_path)

    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    # 启用梯度检查点（如果配置）
    if model_cfg.get("gradient_checkpointing", "false").lower() == "true":
        model.gradient_checkpointing_enable()

    trainer = CustomFamilyTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=custom_data_collator,
        forget_loss=cfg.forget_loss,
        save_step_pattern=cfg.save_step_pattern,
        save_dir=cfg.save_dir,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if cfg.forget_loss == "npo":
        outputs_f_ref_dir = f"{cfg.save_dir}/outputs_f_ref.pt"
        if not os.path.exists(outputs_f_ref_dir):
            ref_model = AutoModelForCausalLM.from_pretrained(
                cfg.model_path,
                config=config,
                use_flash_attention_2=False,
                torch_dtype=torch.bfloat16,
                token=os.environ['HF_TOKEN'],
                trust_remote_code=True
            )
            ref_model.eval()
            ref_model = trainer.e_prepare_deepspeed(ref_model)
            with torch.no_grad():
                inputs = trainer.train_dataset[0]
                input_ids, labels, attention_mask = inputs[0], inputs[1], inputs[2]
                input_ids = input_ids.unsqueeze(0).to(local_rank)
                labels = labels.unsqueeze(0).to(local_rank)
                attention_mask = attention_mask.unsqueeze(0).to(local_rank)
                outputs_f_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
            ref_model.destroy()
            del ref_model
            gc.collect()
            torch.cuda.empty_cache()
            torch.save(outputs_f_ref, outputs_f_ref_dir)
            exit()
        trainer.outputs_f_ref_logits = torch.load(outputs_f_ref_dir).logits.to(local_rank)
    # 开始训练
    trainer.train()

    # 删除所有 "global_step*" 文件夹
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                import shutil
                shutil.rmtree(global_step_dir)

if __name__ == "__main__":
    main()

```

## finetune_reinforced_model.py

```python
from data_module import custom_data_collator, FamilyForgetDataset
from unlearn_trainer import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["model_id"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file
    #if master process
    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    subsample = torch.load(cfg.subsample_path)
    shuffled_unlearn_data_id = int(subsample[cfg.unlearn_data_id])
    torch_format_dataset = FamilyForgetDataset(cfg.data_path, tokenizer=tokenizer, model_configs=model_cfg,max_length=500, unlearn_data_id=shuffled_unlearn_data_id,question_key='question4', answer_key='answer4') 

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    print("max_steps calc parmas : len(torch_format_dataset)", len(torch_format_dataset), "num_epochs:", cfg.num_epochs, "batch_size:", batch_size, "gradient_accumulation_steps:",gradient_accumulation_steps, "num_devices:", num_devices, "steps_per_epoch:",steps_per_epoch)
    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    
    lr = float(model_cfg["reinforce_lr"])
    
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=lr,
            lr_scheduler_type=cfg.lr_scheduler_type,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_steps=max_steps,
            save_strategy="steps",
            save_only_model=True,
            ddp_find_unused_parameters= False,
            evaluation_strategy="no",
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search(r"pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search(r"model-*\.safetensors", file):
            path_found = True
            break

    if path_found:
        print("INSIDE PATTH FOUND")
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, token=os.environ['HF_TOKEN'], trust_remote_code = True)
    
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True)
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()

```

## data_module.py

```python
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index
import os

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
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
        
    encoded_answer = tokenizer(
        new_answer, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
        
        
    #change label to -100 for question tokens
#     print(encoded['input_ids'][num_question_tokens], label[num_question_tokens])
    for i in range(num_question_tokens): label[i] = -100
    
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)
    

class FamilyForgetDataset(Dataset):
    def __init__(self, data_path, tokenizer, model_configs, max_length=512,  unlearn_data_id=0, question_key=None, answer_key=None):
        super(FamilyForgetDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = datasets.Dataset.from_dict(torch.load(data_path))
        self.data = add_dataset_index(self.data)
        self.qk = question_key
        self.ak = answer_key
        self.unlearn_data_id = unlearn_data_id
        self.model_configs = model_configs

    def __len__(self):
        return int(os.environ.get('WORLD_SIZE', 1)) 

    def __getitem__(self, idx):
        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        question = self.data[self.unlearn_data_id][self.qk]
        answers = self.data[self.unlearn_data_id][self.ak]
        indices = self.data[self.unlearn_data_id]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)
    def to_csv(self, output_path: str = 'family_forget_dataset.csv'):
        df = self.data.to_pandas()
        df.to_csv(output_path, index=False)
    
def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


```

## collect_python_codes.py

```python
import os

def collect_python_files(directory):
    """
    遍历指定目录及其子目录，收集所有以 .py 结尾的Python文件路径。
    """
    python_files = []
    for root, dirs, files in os.walk(directory):
        # 忽略隐藏文件夹
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    return python_files

def write_to_markdown(python_files, output_file):
    """
    将收集到的Python文件内容写入到一个Markdown文件中，
    每个文件的内容都包含在一个代码块中，并附有文件名作为标题。
    """
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for file in python_files:
            # 写入文件名作为二级标题
            md_file.write(f'## {os.path.relpath(file)}\n\n')
            md_file.write('```python\n')
            with open(file, 'r', encoding='utf-8') as py_file:
                code = py_file.read()
                md_file.write(code)
            md_file.write('\n```\n\n')

if __name__ == "__main__":
    current_directory = os.getcwd()
    python_files = collect_python_files(current_directory)
    output_markdown = 'python_code_collection.md'
    write_to_markdown(python_files, output_markdown)
    print(f"所有Python代码已被收集到 {output_markdown}")

```

## tv_run.py

```python
import torch
import task_vector
import os
from utils import get_model_identifiers_from_yaml
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run TV unlearn with params.")
    
    parser.add_argument('--unlearn_data_id', type=int, required=True, help="Index of sample to be unlearnt")
    parser.add_argument('--alpha_list', type=str, required=False,default=None, help="alphalist")
    parser.add_argument('--ft_dir', type=str, required=True, help="pretrained model directory")
    parser.add_argument('--reinforced_model_dir', type=str, required=True, help="finetuned model directory on the target fact")
    parser.add_argument('--out_dir', type=str, required=True, help="model directory for saving results")
    parser.add_argument('--model_family', type=str, required=True, help="model family")
    args = parser.parse_args()

    some_ft_model_dir = args.ft_dir
    model_dir = args.ft_dir
    some_reinforced_model_dir = args.reinforced_model_dir
    
    model_cfg = get_model_identifiers_from_yaml(args.model_family)
    alphas_str_list = model_cfg["tv_alpha_list"].split(" ")
    alphas = [float(alpha) for alpha in alphas_str_list]

    for alpha in alphas:
        out_dir = args.out_dir + f"/checkpoint-{alpha}"
        task_vector.unlearn(model_dir, out_dir, some_pt_model_dir=some_ft_model_dir,some_ft_model_dir=some_reinforced_model_dir, alpha=alpha)

if __name__ == "__main__":
    main()

```

## task_vector.py

```python
#borrowed from https://github.com/swj0419/muse_bench/blob/main/baselines/baselines/task_vector.py

from transformers import AutoModelForCausalLM

import torch


def load_model(model_dir: str, **kwargs) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        **kwargs
    )


def compare(model1, model2) -> bool:
    """Compares two models.

    Args:
        model1 (_type_): _description_
        model2 (_type_): _description_

    Returns:
        bool: _description_
    """
    dict1, dict2 = model1.state_dict(), model2.state_dict()
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1.keys():
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True


def unlearn(
    model_dir: str,
    out_dir: str | None = None,
    some_pt_model_dir: str | None = None,
    some_ft_model_dir: str | None = None,
    alpha: float = 1.0
):
    if some_pt_model_dir is None or some_ft_model_dir is None:
        raise ValueError("Task vector (ilharco2023) requires some pretrained & finetuned models!")

    task_vector = TaskVector(
        pretrained_state_dict=load_model(some_pt_model_dir).state_dict(),
        finetuned_state_dict=load_model(some_ft_model_dir).state_dict()
    )

    if not task_vector.is_nonzero():
        raise ValueError("Zero task vector encountered!")

    neg_task_vector = -task_vector
    #print("NEGATIVE VECTOR VALUE: ", type(task_vector))
    
    model = load_model(model_dir)
    new_state_dict = neg_task_vector.apply_to(pretrained_model=model, scaling_coef=alpha, in_place=False)
    del model
    new_model = load_model(model_dir, state_dict=new_state_dict, device_map='auto')

    if out_dir is not None:
        new_model.save_pretrained(out_dir, state_dict=new_state_dict)
    return new_model


class TaskVector():
    def __init__(self,
                 pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None,
                 pretrained_state_dict=None, finetuned_state_dict=None):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        if vector is not None:
            self.vector = vector
        else:
            assert (
                (pretrained_checkpoint is not None and finetuned_checkpoint is not None)
                or
                (pretrained_state_dict is not None and finetuned_state_dict is not None)
            )
            with torch.no_grad():
                if pretrained_state_dict is None:
                    pretrained_state_dict = torch.load(pretrained_checkpoint).state_dict()
                if finetuned_state_dict is None:
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector)

    def is_nonzero(self):
        return any([(self.vector[key] != 0).any() for key in self.vector])

    def apply_to(self, pretrained_model, scaling_coef=1.0, in_place=False):
        """Apply a task vector to a pretrained model."""
        print('scaling_coef:',scaling_coef)
        with torch.no_grad():
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
            #torch.save(new_state_dict, "new_state_dict")
        if in_place:
            pretrained_model.load_state_dict(new_state_dict, strict=False)
        return new_state_dict
```

## evaluate_util.py

```python
from vllm import SamplingParams
import torch.nn.functional as F
import torch
from tqdm import tqdm

def eval_qa_vllm(dataset, model_eval, qk="question", ak="answer", question_start_tag="[INST] ", question_end_tag=" [/INST]", answer_tag=""):
    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    sampling_params = SamplingParams(temperature=0, top_p=0.6, max_tokens=10)
    responses = model_eval.generate(prompts, sampling_params)
    outputs = [response.outputs[0].text for response in responses]
    correct = [data[ak].lower() in output.lower() for data, output in zip(dataset, outputs)]
    return correct, responses

def eval_qa_vllm_whp(dataset, model_eval1,model_eval2, tokenizer,alpha_list, max_new_tokens=3, qk="question", ak="answer", question_start_tag = "[INST] ", question_end_tag = " [/INST]", answer_tag=""):
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id
    
    model_eval1.generation_config.pad_token_id = tokenizer.pad_token_id
    model_eval2.generation_config.pad_token_id = tokenizer.pad_token_id

    prompts = [question_start_tag + data[qk] + question_end_tag for data in dataset]
    outputs_list = [[] for alpha in alpha_list]
    
    for i,prompt in enumerate(tqdm(prompts)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model_eval1.device)
        out1 = model_eval1.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_logits=True, output_scores=True)
        out2 = model_eval2.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, return_dict_in_generate=True, output_logits=True, output_scores=True)

        logits_list = []

        len1 = len(out1.logits)
        len2 = len(out2.logits)
        
        length = max(len1, len2)
        vocab_len = out1.logits[0].shape[1]
        if len1 < length:
            n_pads = length-len1
            zero_tensors = tuple(torch.zeros((1, vocab_len)) for _ in range(n_pads))
            out1.logits = out1.logits + zero_tensors
        elif len2 < length:
            n_pads = length-len2
            zero_tensors = tuple(torch.zeros((1, vocab_len)) for _ in range(n_pads))
            out2.logits = out2.logits + zero_tensors

        out2.logits = tuple(logit.to(model_eval1.device) for logit in out2.logits)
        out1.logits = tuple(logit.to(model_eval1.device) for logit in out1.logits)
        
        prob1_batch = F.softmax(torch.cat(list(out1.logits)), dim=-1)
        prob2_batch = F.softmax(torch.cat(list(out2.logits)), dim=-1)


        for i_alpha, alpha in enumerate(alpha_list):
            logits = prob1_batch - alpha * F.relu(prob2_batch-prob1_batch)
            predicted_token_ids = torch.argmax(logits, dim=-1)  # Shape should be [batch_size, seq_length]
            predicted_texts = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
            outputs_list[i_alpha].append(predicted_texts)
        
    correct_list = [[data[ak].lower() in output.lower() for data, output in zip(dataset, outputs)] for outputs in outputs_list]
    return correct_list, outputs_list

```

## calculate_recall_and_acc.py

```python
import argparse
import os
import torch
import numpy as np
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description='calculate the recall and accucracy')
parser.add_argument('--unlearn_data_id', type=int, default=None, help="id of the fact to unlearn")
parser.add_argument('--input_dir', type=str, default=None, help="directory that saves the rettained knowledge base")
args = parser.parse_args()

class Person:
    def __init__(self):
        self.name = None
        self.gender = gender
        self.father = None
        self.mother = None
        self.children = None
        self.husband = None
        self.wife = None


from copy import deepcopy
class Rule:
    def __init__(self, left_tuples, right_tuple):
        self.left_tuples = left_tuples
        self.right_tuple = right_tuple
        self.num_var = max(max(tup[0], tup[2]) for tup in left_tuples + [right_tuple]) + 1
        
    def get_up_edges_list(self, edge_list, edge_type_list, unlearn_edge, unlearn_edge_type):
        source_type_dict = {}
        type_target_dict = {}
        for edge, edge_type in zip(edge_list, edge_type_list):
            source_type = (edge[0], edge_type)
            if source_type in source_type_dict.keys():
                source_type_dict[source_type].append(edge[1])
            else:
                source_type_dict[source_type] = [edge[1]]
                
            type_target = (edge_type, edge[1])
            if type_target in type_target_dict.keys():
                type_target_dict[type_target].append(edge[0])
            else:
                type_target_dict[type_target] = [edge[0]]
        
        var_value = -np.ones(self.num_var)
        var_value[self.right_tuple[0]] = unlearn_edge[0]
        var_value[self.right_tuple[2]] = unlearn_edge[1]
        
        dc_var_value_list = []
        
        def _get_up_edges_list(cur_var_value):
            if (cur_var_value == -1).sum() == 0:
                for tup in self.left_tuples + [self.right_tuple]:
                    if (cur_var_value[tup[0]], tup[1]) not in source_type_dict.keys():
                        return
                    if cur_var_value[tup[2]] not in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                        return
                
                if not any(np.array_equal(cur_var_value, unique_arr) for unique_arr in dc_var_value_list):
                    dc_var_value_list.append(cur_var_value)
                return
            
            for tup in self.left_tuples:
                if cur_var_value[tup[0]] == -1 and cur_var_value[tup[2]] != -1:
                    if (tup[1], cur_var_value[tup[2]]) in type_target_dict.keys():
                        for potential_tup0_val in type_target_dict[(tup[1], cur_var_value[tup[2]])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[0]] = potential_tup0_val
                            _get_up_edges_list(new_cur_var_value)
                elif cur_var_value[tup[2]] == -1 and cur_var_value[tup[0]] != -1:
                    if (cur_var_value[tup[0]], tup[1]) in source_type_dict.keys():
                        for potential_tup0_val in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[2]] = potential_tup0_val
                            _get_up_edges_list(new_cur_var_value)
        
        _get_up_edges_list(var_value)
        
        up_edges_list = []
        for dc_var_value in dc_var_value_list:
            up_edges = []
            for tup in self.left_tuples:
                up_edges.append((dc_var_value[tup[0]], tup[1], dc_var_value[tup[2]]))
            up_edges_list.append(up_edges)
            
        return up_edges_list
    
    def get_dc_edges_list(self, edge_list, edge_type_list):
        source_type_dict = {}
        type_target_dict = {}
        for edge, edge_type in zip(edge_list, edge_type_list):
            source_type = (edge[0], edge_type)
            if source_type in source_type_dict.keys():
                source_type_dict[source_type].append(edge[1])
            else:
                source_type_dict[source_type] = [edge[1]]
                
            type_target = (edge_type, edge[1])
            if type_target in type_target_dict.keys():
                type_target_dict[type_target].append(edge[0])
            else:
                type_target_dict[type_target] = [edge[0]]
        
        dc_var_value_list = []
        def _get_right_edges_list(cur_var_value):
            if (cur_var_value == -1).sum() == 0:
                for tup in self.left_tuples:
                    if (cur_var_value[tup[0]], tup[1]) not in source_type_dict.keys():
                        return
                    if cur_var_value[tup[2]] not in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                        return
                if not any(np.array_equal(cur_var_value, unique_arr) for unique_arr in dc_var_value_list):
                    dc_var_value_list.append(cur_var_value)
                return
            
            for tup in self.left_tuples:
                if cur_var_value[tup[0]] == -1 and cur_var_value[tup[2]] != -1:
                    if (tup[1], cur_var_value[tup[2]]) in type_target_dict.keys():
                        for potential_tup0_val in type_target_dict[(tup[1], cur_var_value[tup[2]])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[0]] = potential_tup0_val
                            _get_right_edges_list(new_cur_var_value)
                elif cur_var_value[tup[2]] == -1 and cur_var_value[tup[0]] != -1:
                    if (cur_var_value[tup[0]], tup[1]) in source_type_dict.keys():
                        for potential_tup0_val in source_type_dict[(cur_var_value[tup[0]], tup[1])]:
                            new_cur_var_value = deepcopy(cur_var_value)
                            new_cur_var_value[tup[2]] = potential_tup0_val
                            _get_right_edges_list(new_cur_var_value)
        
        for edge, edge_type in zip(edge_list, edge_type_list):
            if edge_type == self.left_tuples[0][1]:
                var_value = -np.ones(self.num_var)
                var_value[self.left_tuples[0][0]] = edge[0]
                var_value[self.left_tuples[0][2]] = edge[1]
                _get_right_edges_list(var_value)
        
        
        new_edge_list = []
        new_edge_type_list = []
        
        for dc_var_value in dc_var_value_list:
            new_edge = (dc_var_value[self.right_tuple[0]], dc_var_value[self.right_tuple[2]])
            new_edge_type = (self.right_tuple[1])
            
            if (dc_var_value[self.right_tuple[0]], self.right_tuple[1]) in source_type_dict.keys():
                if dc_var_value[self.right_tuple[2]] in source_type_dict[(dc_var_value[self.right_tuple[0]], self.right_tuple[1])]:
                    continue
            
            if self.right_tuple[1] in ["husband", "uncle", "father", "brother", "nephew"]:
                if person_list[int(dc_var_value[self.right_tuple[2]])].gender != "male":
                    continue
                if self.right_tuple[1] == "husband" and person_list[int(dc_var_value[self.right_tuple[0]])].gender != "female":
                    continue
                    
            elif self.right_tuple[1] in ["wife", "aunt", "mother", "sister", "niece"]:
                if person_list[int(dc_var_value[self.right_tuple[2]])].gender != "female":
                    continue
                if self.right_tuple[1] == "wife" and person_list[int(dc_var_value[self.right_tuple[0]])].gender != "male":
                    continue
            if dc_var_value[self.right_tuple[0]] == dc_var_value[self.right_tuple[2]]:
                continue
            
            new_edge_list.append(new_edge)
            new_edge_type_list.append(new_edge_type)
            
        return new_edge_list, new_edge_type_list


def check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
    cur_minimal_set = set(list(deepcopy(minimal_set)) + list(range(len(edge_list), len(dc_edge_list))))
    
    new_added_id_list = []
    t = 0
    while len(new_added_id_list) > 0 or t == 0:
        new_added_id_list = []
        t = t + 1
        for cur_unlearn_data_id in cur_minimal_set:
            unlearn_edge = dc_edge_list[cur_unlearn_data_id]
            unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id]
            rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]
            if_deducted = False
            for rule in rule_set_related:
                if if_deducted:
                    break
                up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type)
                for up_edges in up_edges_list:
                    up_edges_if_deducted = True
                    for up_edge in up_edges:
                        ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
                        if ind in cur_minimal_set:
                            up_edges_if_deducted = False
                            break
                    if up_edges_if_deducted:
                        if_deducted = True
                        new_added_id_list.append(cur_unlearn_data_id)
                        break
        for new_added_id in new_added_id_list:
            cur_minimal_set.remove(new_added_id)
    if unlearn_data_id in cur_minimal_set:
        return False
    else:
        return True              
                
    
def get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
    minimal_set = set([])
    minimal_set_unverified = set([unlearn_data_id])

    
    #Find a valid unlearning set expanded from the given unlearning result.
    while len(minimal_set_unverified) >= 1:
#         print(minimal_set_unverified)
        cur_unlearn_data_id = random.sample(sorted(minimal_set_unverified), 1)[0]
        minimal_set_unverified.remove(cur_unlearn_data_id)
        minimal_set.add(cur_unlearn_data_id)

        unlearn_edge = dc_edge_list[cur_unlearn_data_id]
        unlearn_edge_type = dc_edge_type_list[cur_unlearn_data_id]
        rule_set_related = [rule for rule in rule_list if rule.right_tuple[1] == unlearn_edge_type]

        for rule in rule_set_related:
            up_edges_list = rule.get_up_edges_list(dc_edge_list, dc_edge_type_list, unlearn_edge, unlearn_edge_type)
            for up_edges in up_edges_list:
                if_suf = 0
                for up_edge in up_edges:
                    ind = get_edge_id((up_edge[0], up_edge[2]), dc_edge_list)
                    if (ind in minimal_set) or (ind in minimal_set_unverified):
                        if_suf = 1
                        break
                if if_suf == 0:
                    rand_edge = random.sample(up_edges, 1)[0]
                    rand_ind = get_edge_id((rand_edge[0], rand_edge[2]), dc_edge_list)
                    minimal_set_unverified.add(rand_ind)
        
    minimal_set = set([i for i in minimal_set if i < len(edge_list)])
    #Prune the valid unlearning set by removing redundant element from the extended part
    
    C = []
    t = 0
    while len(C) != 0 or t==0:
        C = []
        t = t+1
        shuffled_minimal_set = np.asarray(list(minimal_set))[np.random.permutation(len(minimal_set))]
        for data_id in shuffled_minimal_set:
            minimal_set.remove(data_id)
            if not check_if_in_deductive_closure(unlearn_data_id, minimal_set, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list):
                C.append(data_id)
            else:
                minimal_set.add(data_id)
    return minimal_set

    
def get_prec_rec_acc(minimal_set, unlearn_ind):
    minimal_set_ind = np.zeros(len(unlearn_ind))
    minimal_set_ind[list(minimal_set)] = 1
    prec = (minimal_set_ind * unlearn_ind).sum() / max(unlearn_ind.sum(), 1e-8)
    rec = (minimal_set_ind * unlearn_ind).sum() / minimal_set_ind.sum()
    acc = 1 - (unlearn_ind * (1 - minimal_set_ind)).sum() / (len(unlearn_ind) - len(minimal_set))
    return prec, rec, acc
    
    
def get_valid_unlearn_general(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, unlearn_ind, rule_list, num_seed=10):
    if os.path.exists(f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt"):
        minimal_unlearn_set = torch.load(f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt")
    else:
        minimal_unlearn_list = []
        for seed in tqdm(range(num_seed)):
            minimal_set = get_minimal_nec_unlearn_and_not_included_unlearn(unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, rule_list, seed)
            minimal_unlearn_list.append(minimal_set)
        minimal_unlearn_set = set([frozenset(minimal_set) for minimal_set in minimal_unlearn_list])
        torch.save(minimal_unlearn_set, f"synthetic_data/unlearn_minimal_set/{unlearn_data_id}.pt")
    precision_list = []
    recall_list = []
    acc_list = []
    for minimal_set in minimal_unlearn_set:
        prec, rec, acc = get_prec_rec_acc(minimal_set, unlearn_ind)
        precision_list.append(prec)
        recall_list.append(rec)
        acc_list.append(acc)
    
    return precision_list, recall_list, acc_list, minimal_unlearn_set

def get_edge_id(edge, edge_list):
    for i, _edge in enumerate(edge_list):
        if _edge == edge:
            return i
        
        
def get_deductive_closure(edge_list, edge_type_list, rule_list):
    dc_edge_list, dc_edge_type_list = deepcopy(edge_list), deepcopy(edge_type_list)
    new_edge_list = []
    new_edge_type_list = []
    cur_iter=0
    while len(new_edge_list) > 0 or cur_iter == 0:
        new_edge_list = []
        new_edge_type_list = []
        for rule in rule_list:
            _new_edge_list, _new_edge_type_list = rule.get_dc_edges_list(dc_edge_list, dc_edge_type_list)
            dc_edge_list = dc_edge_list + _new_edge_list
            dc_edge_type_list = dc_edge_type_list + _new_edge_type_list
            
            new_edge_list = new_edge_list + _new_edge_list
            new_edge_type_list = new_edge_type_list + _new_edge_type_list
            
        cur_iter += 1
    return dc_edge_list, dc_edge_type_list
        
(edge_list, edge_type_list, fixed_names, person_list) = torch.load("synthetic_data/family-200-graph.pt")
rule_list = torch.load("synthetic_data/family_rule.pt")
dc_edge_list, dc_edge_type_list = get_deductive_closure(edge_list, edge_type_list, rule_list)
shuffled_edge_id_list = torch.load("synthetic_data/subsample.pt")

shuffled_unlearn_data_id = shuffled_edge_id_list[args.unlearn_data_id]

if args.input_dir is None:
    print("pre-compute the minimal deep unlearning set only")
    precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(shuffled_unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, np.zeros(len(edge_list)), rule_list, num_seed=100)
    exit()
    
rel_ind = np.asarray(torch.load(f"{args.input_dir}/relationships_correct.pt")).astype(np.float32)
unlearn_ind = 1 - rel_ind
bio_ind = torch.load(f"{args.input_dir}/biographies_correct.pt")

precision_list, recall_list, accuracy_list, minimal_unlearn_list = get_valid_unlearn_general(shuffled_unlearn_data_id, edge_list, edge_type_list, dc_edge_list, dc_edge_type_list, unlearn_ind, rule_list, num_seed=100)

rec = max(recall_list)
argmax = np.asarray(recall_list).argmax()
acc_rel = accuracy_list[argmax]
acc_bio = np.asarray(bio_ind).mean()

num_rel = len(rel_ind)
num_bio = len(bio_ind)
size_mul = len(list(minimal_unlearn_list)[argmax])
acc_all = ((acc_bio * num_bio) + accuracy_list[argmax] * ( num_rel - size_mul)) / (num_bio + num_rel - size_mul)
print(("recall", "accuracy of relationships", "accuracy of biographies", "accurcy of all knowledge base"))
print((rec, acc_rel, acc_bio, acc_all))
torch.save((rec, acc_rel, acc_bio, acc_all), f"{args.input_dir}/rec_acc.pt")
```

## utils.py

```python
import yaml
import copy
import numpy as np
from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted
def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset
```

## unlearn_trainer.py

```python
import torch
from transformers import Trainer
import torch.nn.functional as F
import os
import copy
import numpy as np

import deepspeed
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
    is_deepspeed_available,
)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


class CustomFamilyTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop("forget_loss")
        self.save_dir = kwargs.pop("save_dir")
        self.save_step_pattern = kwargs.pop("save_step_pattern")
        super(CustomFamilyTrainerForgetting, self).__init__(*args, **kwargs)

        if self.loss_type == "npo":
            self.beta = 0.1
            self.outputs_f_ref_logits = None

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_type == "ga":
            forget_inputs = inputs
            input_ids, labels, attention_mask = inputs
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "npo":
            forget_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs

            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            neg_log_ratio = self.outputs_f_ref_logits - outputs.logits
            print(neg_log_ratio)
            loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self, model, inputs, prediction_loss_only: bool, ignore_keys=None
    ):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        curr_step = self.state.global_step
        if self.save_step_pattern == "log":
            import math

            if curr_step not in [1, 2, 4, 8, 16, 32, 64]:
                return

        print(f"Saving model at step {curr_step}")

        curr_save_dir = os.path.join(self.save_dir, f"checkpoint-{curr_step}")
        self.save_model(curr_save_dir)

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        # set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False

        return model

```

## whp.py

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
from pathlib import Path
from utils import get_model_identifiers_from_yaml
import argparse
import gc
import numpy as np
from vllm.distributed.parallel_state import destroy_model_parallel
from datasets import Dataset
import argparse
from evaluate_util import eval_qa_vllm_whp


parser = argparse.ArgumentParser(description='evaluate whp by vllm')
parser.add_argument('--curr_save_dir_top', type=str, default=None, help="directory to save results")
parser.add_argument('--model_dir', type=str, default=None, help="pretrained model directory")
parser.add_argument('--reinforced_model_dir', type=str, default=None, help="finetuned model directory on the target fact")
parser.add_argument('--unlearn_data_id', type=int, default=None, help="id of the fact to unlearn")
parser.add_argument('--model_family', type=str, default=None, help="model family")
parser.add_argument('--max_new_tokens', type=int, default=10, help="max new tokens to be generated")

args = parser.parse_args()

torch.cuda.empty_cache()

model_family = args.model_family
model_dir= args.model_dir
reinforced_model_dir = args.reinforced_model_dir
unlearn_data_id = args.unlearn_data_id
curr_save_dir_top = args.curr_save_dir_top
max_new_tokens = args.max_new_tokens

model_cfg = get_model_identifiers_from_yaml(model_family)
model_id = model_cfg['model_id']

config = AutoConfig.from_pretrained(model_id)
device_map = "auto"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'], padding_side="left")

model_eval1 = AutoModelForCausalLM.from_pretrained(model_dir, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16,trust_remote_code = True, token=os.environ['HF_TOKEN'], device_map=device_map)
model_eval1.eval()

model_eval2 = AutoModelForCausalLM.from_pretrained(reinforced_model_dir, config=config,use_flash_attention_2=model_cfg["flash_attention2"]=="true",  torch_dtype=torch.bfloat16,trust_remote_code = True, token=os.environ['HF_TOKEN'], device_map=device_map)
model_eval2.eval()

tokenizer.pad_token = tokenizer.eos_token

eval_dataset_list = [Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt")), Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt"))]
eval_dataset_name_list = ["relationships_", "biographies_"]

alphas_str_list = model_cfg["whp_alpha_list"].split(" ")
alphas = [float(alpha) for alpha in alphas_str_list]
# for alpha in alphas:
for eval_dataset, eval_dataset_name in zip(eval_dataset_list, eval_dataset_name_list):
    with torch.no_grad():
        print('Starting Dataset:', eval_dataset_name)
        
        correct_rephrase_list, responses_rephrase_list = eval_qa_vllm_whp(eval_dataset, model_eval1,model_eval2, tokenizer, alphas, max_new_tokens=max_new_tokens, qk="question4", ak="answer4", question_start_tag=model_cfg["question_start_tag"], question_end_tag=model_cfg["question_end_tag"], answer_tag="")

        for alpha, correct_rephrase, responses_rephrase in zip(alphas, correct_rephrase_list, responses_rephrase_list):
            print('Running for alpha:', alpha)
            curr_save_dir = curr_save_dir_top+f'/checkpoint-{alpha}'
            Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
            torch.save(correct_rephrase, f"{curr_save_dir}//{eval_dataset_name}correct.pt")
            torch.save(responses_rephrase, f"{curr_save_dir}//{eval_dataset_name}responses.pt")
            acc = np.asarray(correct_rephrase).astype(np.float32).mean()
            print(f"Accuracy: {acc}")

        gc.collect()
        torch.cuda.empty_cache()


destroy_model_parallel()
del model_eval1
del model_eval2
gc.collect()
torch.cuda.empty_cache()
```

## vllm_eval.py

```python
import argparse
import datasets
import gc
import torch
import numpy as np
from vllm import LLM
from vllm.distributed.parallel_state import destroy_model_parallel
from pathlib import Path
from datasets import Dataset

from utils import get_model_identifiers_from_yaml
from evaluate_util import eval_qa_vllm

parser = argparse.ArgumentParser(description='evaluate llm by vllm')
parser.add_argument('--curr_save_dir', type=str, default=None)
parser.add_argument('--model_family', type=str, default="llama2-7b")
parser.add_argument('--clean_cache', type=str, default="false")
args = parser.parse_args()

curr_save_dir = args.curr_save_dir
model_cfg = get_model_identifiers_from_yaml(args.model_family)
model_id = model_cfg["model_id"]

#load vllm model
model_eval = LLM(curr_save_dir, tokenizer=model_id, device="auto", dtype = "half",tensor_parallel_size=8)
# eval_dataset = datasets.load_from_disk(curr_save_dir+"/eval.hf")

# eval_dataset_list = [Dataset.from_dict(torch.load("synthetic_data/family_relationships.pt")), Dataset.from_dict(torch.load("synthetic_data/family_biographies.pt"))]
# eval_dataset_name_list = ["relationships_", "biographies_"]

full_dataset = Dataset.from_dict(torch.load("data/mhu_01_forget_dict.pt"))
eval_dataset_list = [full_dataset]
eval_dataset_name_list = ["forget_set"]

#remove local model
if args.clean_cache == "true":
    import shutil
    shutil.rmtree(curr_save_dir)

Path(curr_save_dir).mkdir(parents=True, exist_ok=True)

for eval_dataset, eval_dataset_name in zip(eval_dataset_list, eval_dataset_name_list):
    with torch.no_grad():
        correct, responses = eval_qa_vllm(eval_dataset, model_eval, qk='single_hop_question', ak='single_hop_answer', question_start_tag=model_cfg["question_start_tag"], question_end_tag=model_cfg["question_end_tag"], answer_tag=model_cfg["answer_tag"])
        torch.save(correct, f"{curr_save_dir}/{eval_dataset_name}correct.pt")
        torch.save(responses, f"{curr_save_dir}/{eval_dataset_name}responses.pt")
        acc = np.asarray(correct).astype(np.float32).mean()
        print(f"{eval_dataset}accuracy: {acc}")

destroy_model_parallel()
del model_eval
gc.collect()
torch.cuda.empty_cache()
```

## common_dataset.py

```python
import torch
from torch.utils.data import Dataset
import datasets
from utils import get_model_identifiers_from_yaml, add_dataset_index
import os

def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs, unlearning_target):
    question_start_token = model_configs.get('question_start_tag', "<Q>")
    question_end_token = model_configs.get('question_end_tag', "</Q>")
    answer_token = model_configs.get('answer_tag', "<A>")
    
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    
    full_text = unlearning_target
    # print('FULL TEXT:',full_text)
    print(f"FULL TEXT: {full_text}")
    
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
        answer = sample[self.ak]
        unlearning_target = sample['unlearning_target']
        # indices = sample.get('index', idx)  # 如果没有 'index' 字段，使用 idx 作为索引
        indices = [idx]

        # if isinstance(answers, str):
        #     answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        # for answer in answers:
        #     converted_data = convert_raw_data_to_model_format(
        #         self.tokenizer, 
        #         self.max_length, 
        #         question=question, 
        #         answer=answer, 
        #         model_configs=self.model_configs
        #     )
        #     pad_input_ids_list.append(converted_data[0])
        #     label_list.append(converted_data[1])
        #     pad_attention_mask_list.append(converted_data[2])
        
        converted_data = convert_raw_data_to_model_format(
            self.tokenizer, 
            self.max_length, 
            question=question, 
            answer=answer, 
            model_configs=self.model_configs,
            unlearning_target=unlearning_target
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

```

## multi-hop-unlearning/partition.py

```python
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


```

## multi-hop-unlearning/statis.py

```python
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
```

## multi-hop-unlearning/test.py

```python
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
```

## multi-hop-unlearning/preprocessing.py

```python
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
```

