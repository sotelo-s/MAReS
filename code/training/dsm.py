'''
    Zero-shot LLM experiments, using criteria and questionnaire data from DSM-5 and ASRS-5, respectively.
'''

import pandas as pd
import torch
from transformers import pipeline, logging, AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from sklearn.model_selection import train_test_split
import numpy as np
import os
import gc
import datetime
import json, re


logging.set_verbosity_error()

MIN_CONTENT = 15
MIN_WORDS = 30
MAX_TOKENS = 400
SEED_VALUE = 22

torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed_all(SEED_VALUE)


in_content_file = "../../in/content.csv"
in_users_file = "../../in/users.csv"
out_folder = "../../out/experiments/llm"
dsm5_file = "./data/dsm5_adhd_criteria.txt"
asrs_file = "./data/asrs.json"
dsm5_prompt_file = "./data/dsm5_prompt.txt"
asrs5_prompt_file = "./data/asrs5_prompt.txt"

with open(dsm5_file, 'r',encoding="utf-8") as file:
    dsm5_criteria = file.read()

with open(asrs_file, 'r',encoding="utf-8") as file:
    questions_asrs5 = json.load(file)["asrs-5"]
    
with open(dsm5_prompt_file, 'r',encoding="utf-8") as file:
    DSM5_BASE_PROMPT = file.read()
    
with open(asrs5_prompt_file, 'r',encoding="utf-8") as file:
    ASRS5_BASE_PROMPT = file.read()

if not os.path.exists(f"{out_folder}"):
    os.makedirs(f"{out_folder}")

models = {
    "llava":"llava-hf/llava-v1.6-mistral-7b-hf",
    "qwen":"Qwen/Qwen3-VL-8B-Instruct"
}

summary_name = "Falconsai/text_summarization"
tokenizer = AutoTokenizer.from_pretrained(summary_name)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_name)

quant_conf = BitsAndBytesConfig(load_in_8bit=True)

content_df = pd.read_csv(in_content_file)
users_df = pd.read_csv(in_users_file)


def run_experiment(test_users,val_users, name, model,type="question",summary=False, version=1):
    start = datetime.datetime.now()

    pipe = pipeline(task="image-text-to-text",
                        model=model,
                        device=0 if torch.cuda.is_available() else -1,
                        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        model_kwargs={"quantization_config": quant_conf} if torch.cuda.is_available() else {},
    )


    if type == "criteria":
        func = dsm5_label
    else:
        func = questionnaire
        


    #validacion
    val_results = val_users.apply(func,pipe=pipe,summary=summary,axis=1,result_type="expand")
    if type=="question":
        val_results["predicted_probability"] = (val_results.filter(like="value_question_").mean(axis=1) -1)/4
    val_results["dataset"] = "validation"

    #test
    test_results = test_users.apply(func,pipe=pipe,summary=summary,axis=1,result_type="expand")
    if type=="question":
        test_results["predicted_probability"] = (test_results.filter(like="value_question_").mean(axis=1) -1)/4
    test_results["dataset"] = "test"

    end = datetime.datetime.now()
    print(f"Total runtime of experiment ({name}) is: {end - start}")

    #guardo
    datat_content = pd.concat([val_results, test_results], axis=0)
    datat_content["experiment_type"] = "llm"
    datat_content["experiment_name"] = name
    datat_content["version"] = version

    if not os.path.exists(f"{out_folder}/{name}"):
        os.makedirs(f"{out_folder}/{name}")

    datat_content.to_csv(f"{out_folder}/{name}/users_predictions.csv",index=False)

    del pipe    
    gc.collect()


def dsm5_label(row,pipe,summary):
    user_data = {
                "user":row["user"],
                "true_label":row["has_ADHD"]
    }

    if summary:
        context = summarize("\n".join(no_adhd_text[no_adhd_text["user"]==row["user"]]["text"]))
    else:
        context = reduce_text(pipe,"\n".join(no_adhd_text[no_adhd_text["user"]==row["user"]]["text"]))

    
    prompt = DSM5_BASE_PROMPT.format(dsm5_criteria=dsm5_criteria,context=context)
    

    try:
        response = pipe(
            text=[{"role":"user","content":[{"type": "text", "text":prompt}]}],
            do_sample=False,
            return_full_text=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
            clean_up_tokenization_spaces=False,
            truncation=True,
            max_length=3000,
            max_new_tokens=3
        )
        answer = response[0]['generated_text'].strip().upper()
        clean_answer = extract_yes_no(answer)

        if "1" in clean_answer:
            user_data["predicted_probability"] = 1.0
        elif "0" in clean_answer:
            user_data["predicted_probability"] = 0.0
        else:
            user_data["predicted_probability"] = 0.5

        user_data["gen_label"] = answer
        
        
    
    except Exception as e:
        print(f"Error processing: {e}")
        user_data["predicted_probability"] = 0.5

    return user_data


    

def questionnaire(row,pipe,summary):
    user_data = {
            "user":row["user"],
            "true_label":row["has_ADHD"]
        }

    if summary:
        context = summarize("\n".join(no_adhd_text[no_adhd_text["user"]==row["user"]]["text"]))
    else:
        context = reduce_text(pipe,"\n".join(no_adhd_text[no_adhd_text["user"]==row["user"]]["text"]))


    for index,question in enumerate(questions_asrs5):
        '''
        Uses a LLM and a questionnaire to determine if user has ADHD.
        '''
        
        prompt = ASRS5_BASE_PROMPT.format(context=context,question=question)
    
    
        try:

            
            
            response = pipe(
                prompt,
                do_sample=False,
                return_full_text=False,
                pad_token_id=pipe.tokenizer.eos_token_id,
                clean_up_tokenization_spaces=False,
                truncation=True,
                max_length=2048,
                max_new_tokens=3
            )
            answer = response[0]['generated_text'].strip()

            print(answer)

            
            user_data[f"value_question_{index}"] = extract_rating(answer)
            user_data[f"gen_question_{index}"] = answer
        
        
        except Exception as e:
            print(f"Error processing: {e}")
            user_data["predicted_probability"] = 0.5
    
    return user_data

def extract_rating(text):
    matches = re.findall(r'\b([1-5])\b',text)
    return int(matches[-1]) if matches else 3

def extract_yes_no(text):
    match = re.search(r'\b(1|0)\b',text)
    return match.group(1) if match else "MAYBE"

def summarize(text):
    #return summarizer(text, max_length=1000,do_sample=False)[0]["summary_text"]
    inputs = tokenizer(text,return_tensors="pt",max_length=1024,truncation=True)
    summary_ids = summary_model.generate(
        inputs["input_ids"],
        max_length=130,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0],skip_special_tokens=True)

def reduce_text(pipe,text):
    tokens = pipe.tokenizer.encode(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        return pipe.tokenizer.decode(tokens,skip_special_tokens=True)


#descartamos las que tienen pocas palabras/contenido
users_df = users_df[users_df["n_content"] >= MIN_CONTENT]
content_df = content_df[content_df["word_count"] >= MIN_WORDS]

#eliminar texto que tenga patron TDAH
no_adhd_text = content_df.drop(content_df[content_df["has_ADHD_pattern"] == True].index)#[:200]

#elimino vacias
no_adhd_text = no_adhd_text.replace('', np.nan)
no_adhd_text = no_adhd_text[~no_adhd_text["text"].str.isspace()]

no_adhd_text = no_adhd_text.dropna(subset=['text'])
no_adhd_text = no_adhd_text[no_adhd_text['text'].astype(str).str.strip() != '']
no_adhd_text = no_adhd_text[no_adhd_text['text'].astype(str).str.len() > 0]

no_adhd_text['text'] = no_adhd_text['text'].astype(str)

join_data = users_df.merge(no_adhd_text,left_on="id",right_on="user",how="inner")

user_labels = join_data[['user', 'has_ADHD']].drop_duplicates()

#train test val 70 15 15
train_users, temp_users = train_test_split(
    user_labels,
    test_size=0.30,
    random_state=SEED_VALUE,
    stratify=user_labels["has_ADHD"]
)

val_users, test_users = train_test_split(
    temp_users,
    test_size=0.50,
    random_state=SEED_VALUE,
    stratify=temp_users["has_ADHD"]
)

train = no_adhd_text[no_adhd_text['user'].isin(train_users['user'])]
val = no_adhd_text[no_adhd_text['user'].isin(val_users['user'])]
test = no_adhd_text[no_adhd_text['user'].isin(test_users['user'])]


for m_name,model in models.items():
    run_experiment(test_users,val_users,f"{m_name}_userlevel_questionnaire_textonly_nosummary",model,type="question",summary=False,version=1)
    run_experiment(test_users,val_users,f"{m_name}_userlevel_questionnaire_textonly_summary",model,type="question",summary=True,version=1)
    run_experiment(test_users,val_users,f"{m_name}_userlevel_criteria_textonly_nosummary",model,type="criteria",summary=False,version=1)
    run_experiment(test_users,val_users,f"{m_name}_userlevel_criteria_textonly_summary",model,type="criteria",summary=True,version=1)

