'''
    Generates ground truth from 30% of users with an LLM.
'''

import pandas as pd
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig
from sklearn.model_selection import train_test_split

model_name = "google/gemma-3-12b-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoProcessor.from_pretrained(model_name)


model = Gemma3ForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"":0},
    torch_dtype=torch.bfloat16,
)

in_content_file = "../../in/content.csv"
in_users_file = "../../in/users.csv"
out_users_file = "../../out/users_model_gt.csv"
out_content_file = "../../out/content_model_gt.csv"

SEED_VALUE = 22

sarcasm_thr = 0.7 #If user has >0.7 prob of being sarcastic, label = 0

content_df = pd.read_csv(in_content_file)
users_df = pd.read_csv(in_users_file)

@torch.inference_mode()
def model_label(row):
    
    prompt = f"""Classify if the SPEAKER asserts they personally have ADHD.

YES examples: "I have ADHD", "I was diagnosed with ADHD", "My ADHD brain..."
NO examples: "my friend has ADHD", "I don't have ADHD", "I might have ADHD", "do I have ADHD?", "if I have ADHD then..."

Text: "{row['text']}"

Answer with exactly one word: YES or NO.
Answer:"""

    try:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type":"text","text":prompt}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        inputs = inputs.to(model.get_input_embeddings().weight.device)


        outputs = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            #pad_token_id=tokenizer.eos_token_id,
        )

        input_length = inputs["input_ids"].shape[-1]
        response = tokenizer.decode(
            outputs[0, input_length:],
            skip_special_tokens=True,
        ).strip().upper()        
        
        answer = response.split()[0].strip(".,!:;") if response else ""

        if answer == "YES":
            row["has_ADHD_pattern_model"] = True
        elif answer == "NO":
            row["has_ADHD_pattern_model"] = False
        else:
            row["has_ADHD_pattern_model"] = None
        
    except Exception as e:
        print(f"Error processing: {e}")
        row["has_ADHD_pattern_model"] = None

    return row

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


_, users_20 = train_test_split(
    users_df,
    test_size=0.30,
    random_state=SEED_VALUE,
    stratify=users_df["has_ADHD"]
)

content_20 = content_df[content_df["user"].isin(users_20["id"])].copy()

content_20["text"] = content_20["text"].astype(str).fillna("")
content_20 = content_20.apply(model_label,axis=1)
content_20 = content_20.apply(sarcasm_label,axis=1)

users_20 = users_20.apply(update_users,axis=1)

content_20.to_csv(out_content_file,index=False)
users_20.to_csv(out_users_file,index=False)
