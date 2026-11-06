'''
    Generates ground truth from 30% of users with an LLM.
'''

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re
import math

model_name = "openai/gpt-oss-20b"
BATCH_SIZE = 4


tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map={"":0},
)
model.eval()


generation_config = GenerationConfig(
    max_new_tokens=256,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)

in_content_file = "../../in/content.csv"
in_users_file = "../../in/users.csv"
out_users_file = "../../out/users_gpt.csv"
out_content_file = "../../out/content_gpt.csv"
user_val_file = "../../out/experiments/val_users.csv"
user_test_file = "../../out/experiments/test_users.csv"
checkpoint_file = "../../out/content_gpt_checkpoint.csv"
prompt_file = "./data/annotation_prompt.txt"

SEED_VALUE = 22

sarcasm_thr = 0.7 #If user has >0.7 prob of being sarcastic, label = 0

content_df = pd.read_csv(in_content_file)
users_df = pd.read_csv(in_users_file)

with open(prompt_file, 'r') as file:
    SYSTEM_PROMPT = file.read()


def build_prompt(text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role":"user","content":f"Classify this text:\n<text>\n{text}\n</text>"}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    
def parse_answer(raw):
    match = re.search(
        r"final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        raw,
        re.DOTALL,
    )
    answer = match.group(1).strip() if match else raw.strip()

    match = re.match(r"^\s*(TRUE|FALSE)\b", answer, re.IGNORECASE)
    if match:
        return match.group(1).upper() == "TRUE", answer
    return None, answer

@torch.inference_mode()
def label_texts(texts, batch_size=BATCH_SIZE):
    import os

    n = len(texts)

    if os.path.exists(checkpoint_file):
        checkpoint = pd.read_csv(checkpoint_file)

        if len(checkpoint) != n:
            raise ValueError(
                f"Checkpoint has {len(checkpoint)} rows, expected {n}. "
                "Check that the input dataset and row order have not changed."
            )

        labels = checkpoint["has_ADHD_pattern_model"].tolist()
        labels = [None if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in labels]

        answers = checkpoint["generated_answer"].fillna("").tolist()

        done = checkpoint["has_ADHD_pattern_model"].notna().tolist()

        print(f"Resuming: {sum(done)}/{n} rows completed", flush=True)
    else:
        labels = [None] * n
        answers = [""] * n
        done = [False] * n

    pending = [i for i in range(n) if not done[i]]

    for start in range(0, len(pending), batch_size):
        indices = pending[start:start + batch_size]
        batch_texts = [str(texts[i]) for i in indices]
        prompts = [build_prompt(text) for text in batch_texts]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        input_device = model.get_input_embeddings().weight.device
        enc = {k: v.to(input_device) for k, v in enc.items()}

        outputs = model.generate(
            **enc,
            generation_config=generation_config,
        )

        generated = outputs[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(
            generated, skip_special_tokens=False
        )

        for i, raw in zip(indices, decoded):
            label, answer = parse_answer(raw)
            labels[i] = label
            answers[i] = answer

        checkpoint = content_20.copy()
        checkpoint["has_ADHD_pattern_model"] = labels
        checkpoint["generated_answer"] = answers
        checkpoint.to_csv(checkpoint_file, index=False)

        completed = sum(x is not None for x in labels)
        print(
            f"Processed {completed}/{n} rows; "
            f"checkpoint saved to {checkpoint_file}",
            flush=True,
        )

    return labels, answers



def sarcasm_label(row):
    '''
    Takes the sarcasm prob. and pattern recognition into account to determine if user says they have ADHD.
    '''
    row["has_ADHD_pattern_model_sarcasm"] = row["has_ADHD_pattern_model"] and (row["sarcasm_prob"] < sarcasm_thr )
    return row
    

def update_users(row):
    '''
    Adds new "has_ADHD" attributes according to model and sarcasm.
    '''
    row["has_ADHD_model"] = content_20.loc[(content_20["user"]==row["id"]),"has_ADHD_pattern_model"].any()
    row["has_ADHD_model_sarcasm"] = content_20.loc[(content_20["user"]==row["id"]),"has_ADHD_pattern_model_sarcasm"].any()
    return row


user_ids = set(pd.concat([pd.read_csv(user_val_file),pd.read_csv(user_test_file)])["user"])
users_20 = users_df[users_df["id"].isin(user_ids)]

content_20 = content_df[content_df["user"].isin(users_20["id"])].copy()

content_20["text"] = content_20["text"].fillna("").astype(str)
labels, answers = label_texts(content_20["text"].tolist(),batch_size=BATCH_SIZE)

content_20["has_ADHD_pattern_model"] = labels
content_20["generated_answer"] = answers


content_20 = content_20.apply(sarcasm_label,axis=1)
users_20 = users_20.apply(update_users,axis=1)

content_20.to_csv(out_content_file,index=False)
users_20.to_csv(out_users_file,index=False)
