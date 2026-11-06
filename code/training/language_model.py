'''
    Zero-shot LLM experiments.
'''

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import random
import datetime
import gc
from pathlib import Path
from PIL import Image
from transformers import pipeline, BitsAndBytesConfig
import re



user_file = "../../in/users.csv"
content_file = "../../in/content.csv"
images_folder = "../../in/media"
out_folder = "../../out/experiments/llm"
prompt_text_file = "./data/direct_textonly_prompt.txt"
prompt_images_file = "./data/direct_withimages_prompt.txt"

with open(prompt_text_file,"r",encoding="utf-8") as file:
    TEXTONLY_BASE_PROMPT = file.read()
    
with open(prompt_images_file,"r",encoding="utf-8") as file:
    WITHIMAGES_BASE_PROMPT = file.read()


MIN_CONTENT = 15
MIN_WORDS = 30
MAX_IMAGES = 500
MAX_TOKENS = 400

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

SEED_VALUE = 22

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed_all(SEED_VALUE)

quant_conf = BitsAndBytesConfig(load_in_8bit=True)

models = {
    "llava":"llava-hf/llava-v1.6-mistral-7b-hf",
    "qwen":"Qwen/Qwen3-VL-8B-Instruct"
}

'''
FIELD_NAMES = [
    'user'/'content',
    'experiment_type',
    'experiment_name',
    'dataset',
    'true_label',
    'predicted_probability'
    'version',
]'''

def reduce_text(pipe,text):
    tokens = pipe.tokenizer.encode(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        return pipe.tokenizer.decode(tokens,skip_special_tokens=True)

def reduce_text(pipe,text):
    tokens = pipe.tokenizer.encode(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        return pipe.tokenizer.decode(tokens,skip_special_tokens=True)

def classify_users(data,prompt_type,pipe,all_in_one_prompt,include_images,only_text_with_images):
    #prompt type pro ahora no se utiliza
    results = []
    idx=0
    
    if all_in_one_prompt:
        for user_id in data["id_x"].unique():
            user_data = data[data["id_x"]==user_id]
            texts = reduce_text(pipe,"\n".join(user_data["text"]))
            
            prompt = TEXTONLY_BASE_PROMPT.format(texts=texts)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":prompt}
                    ]
                }
            ]
            
            output = pipe(
                messages,
                do_sample=False,
                return_full_text=False,
                pad_token_id=pipe.tokenizer.eos_token_id,
                clean_up_tokenization_spaces=False,
                #truncation=True,
                #max_length=3000,
                max_new_tokens=3
            )

        
            
            gen = output[0]['generated_text'].strip().upper()

            answer = extract_yes_no(gen)

            if "1" in answer:
                label = 1.0
            elif "0" in answer:
                label = 0.0 
            else:
                label = 0.5
    
            
            results.append({
                'user': user_id,
                'predicted_probability': label,
                'true_label': user_data["has_ADHD"].iloc[0],
                'gen_label' : gen
            })
            
        return pd.DataFrame(results)

    for row in data.itertuples(index=False):
        messages = []
        idx+=1
        text = row.text
        has_media = row.has_media
        
        images = None
        content_path = f"{images_folder}/{row.id_y}"
        path_obj = Path(content_path)

        if  include_images and has_media and path_obj.exists():
            images = [str(image.resolve()) for image in list(path_obj.iterdir())[:MAX_IMAGES]]
        else:
            images = None
            
        open_images = []
        if images:
            for path in images:
                try:
                    image = Image.open(path).convert("RGB")
                    #image.thumbnail((128, 128)) #resize
                    open_images.append(image)
                    
                except Exception as e:
                    ...
            
        #solo analizamos las que tienen imagenes si se marca opcion
        if not open_images:
            if only_text_with_images:
                continue
            
            prompt = TEXTONLY_BASE_PROMPT.format(text=text)
            
        else:
            prompt = WITHIMAGES_BASE_PROMPT.format(text=text)
            
        messages = [
            {
                "role": "user",
                "content": 
                    [{"type": "image","image":image} for image in open_images]
                    +
                    [{
                        "type": "text",
                        "text": prompt
                    }]
            }
        ]

        output = pipe(
                    messages,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=pipe.tokenizer.eos_token_id,
                    clean_up_tokenization_spaces=False,
                    #truncation=True,
                    #max_length=3000,
                    max_new_tokens=3
                )
            
        gen = output[0]['generated_text'].strip().upper()
        answer = extract_yes_no(gen)
        
        if "1" in answer:
            label = 1.0
        elif "0" in answer:
            label = 0.0 
        else:
            label = 0.5
        

        results.append({
            'user': row.id_x,
            'content': row.id_y,
            'predicted_probability': label,
            'true_label': row.has_ADHD,
            'gen_label' : gen
        })
        
    
    return pd.DataFrame(results)

def extract_yes_no(text):
    match = re.search(r'\b(1|0)\b',text)
    return match.group(1) if match else "MAYBE"

def run_experiment(data_train,data_val,data_test,name, prompt_type, model, all_in_one_prompt=False, include_images=False, only_text_with_images=False, version=1):
    #all_in_one_prompt incompatible con include_images/only_text_with_images
    #data train no se utiliza
    start = datetime.datetime.now()
    
    pipe = pipeline(task="image-text-to-text",
                    model=model,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    model_kwargs={"quantization_config": quant_conf} if torch.cuda.is_available() else {},
                    #max_length=2048
    )
    
    val_pred = classify_users(data_val,prompt_type,pipe,all_in_one_prompt,include_images,only_text_with_images)
    test_pred = classify_users(data_test,prompt_type,pipe,all_in_one_prompt,include_images,only_text_with_images)
    
    end = datetime.datetime.now()
    print(f"Total runtime of experiment ({name}) is: {end - start}")
    
    if not os.path.exists(f"{out_folder}/{name}"):
        os.makedirs(f"{out_folder}/{name}")
    
    if all_in_one_prompt == True:
        #no hay predicciones a nivel de contenido
        val_pred["dataset"] = "validation"
        test_pred["dataset"] = "test"
        
        datat_content = pd.concat([val_pred, test_pred], axis=0)
        datat_content["experiment_type"] = "llm"
        datat_content["experiment_name"] = name
        datat_content["version"] = version
        
        datat_content.to_csv(f"{out_folder}/{name}/users_predictions.csv",index=False)
    
    else:
        #A NIVEL DE USUARIO
        #val
        user_val = val_pred.groupby('user').agg(
                    predicted_probability=('predicted_probability', 'mean'),
                    true_label=('true_label', 'first'),
                ).reset_index()
        
        val_to_save = user_val[["user","predicted_probability","true_label"]].copy()
        val_to_save['dataset'] = 'validation'
        
        #test
        user_test = test_pred.groupby('user').agg(
                    predicted_probability=('predicted_probability', 'mean'),
                    true_label=('true_label', 'first'),
                ).reset_index()
        
        test_to_save = user_test[["user","predicted_probability","true_label"]].copy()
        test_to_save['dataset'] = 'test'
        
        datat_content = pd.concat([val_to_save, test_to_save], axis=0)
        datat_content["experiment_type"] = "llm"
        datat_content["experiment_name"] = name
        datat_content["version"] = version
        
        datat_content.to_csv(f"{out_folder}/{name}/users_predictions.csv",index=False)
    
        
        
        #A NIVEL DE CONTENIDO
        #val
        val_to_save = val_pred.drop(columns=["user"]).copy()
        val_to_save['dataset'] = 'validation'
        
        #test
        test_to_save = test_pred.drop(columns=["user"]).copy()
        test_to_save['dataset'] = 'test'
        
        datat_content = pd.concat([val_to_save, test_to_save], axis=0)
        datat_content["experiment_type"] = "llm"
        datat_content["experiment_name"] = name
        datat_content["version"] = version
        
        datat_content.to_csv(f"{out_folder}/{name}/content_predictions.csv",index=False)
    
    

    
    del pipe
    gc.collect()
    

users_df = pd.read_csv(user_file)
content_df = pd.read_csv(content_file)
join_data = users_df.merge(content_df,left_on="id",right_on="user",how="left")

#descartamos las que tienen pocas palabras/contenido
join_data = join_data[join_data["n_content"] >= MIN_CONTENT]
join_data = join_data[join_data["word_count"] >= MIN_WORDS]

#eliminar texto que tenga patron TDAH
no_adhd_text = join_data.drop(join_data[join_data["has_ADHD_pattern"] == True].index)#[:200]

#elimino vacias
no_adhd_text = no_adhd_text.replace('', np.nan)
no_adhd_text = no_adhd_text[~no_adhd_text["text"].str.isspace()]

no_adhd_text = no_adhd_text.dropna(subset=['text'])
no_adhd_text = no_adhd_text[no_adhd_text['text'].astype(str).str.strip() != '']
no_adhd_text = no_adhd_text[no_adhd_text['text'].astype(str).str.len() > 0]

no_adhd_text['text'] = no_adhd_text['text'].astype(str)

user_labels = no_adhd_text[['user', 'has_ADHD']].drop_duplicates()



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


experiments = []
for m_name,model in models.items():
    experiments.append({
        "name": f"llm_{m_name}_mulprompt_textonly",
        "model":model,
        "all_in_one_prompt" : False,
        'include_images': False,
        'only_text_with_images':False,
        "prompt":"zs"
    })
    experiments.append({
        "name": f"llm_{m_name}_mulprompt_withimages",
        "model":model,
        "all_in_one_prompt" : False,
        'include_images': True,
        'only_text_with_images':False,
        "prompt":"zs"
    })
    experiments.append({
        "name": f"llm_{m_name}_oneprompt_textonly",
        "model":model,
        "all_in_one_prompt" : True,
        'include_images': False,
        'only_text_with_images':False,
        "prompt":"zs"
    })


for exp in experiments:
    run_experiment(train,val,test,exp["name"],exp["prompt"],exp["model"],exp["all_in_one_prompt"],exp["include_images"],exp["only_text_with_images"],3)
    
