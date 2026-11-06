'''
    BERT-family experiments.
'''

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding, EvalPrediction
from datasets import DatasetDict, Dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score, accuracy_score
import torch
import random
import datetime



user_file = "../../in/users.csv"
content_file = "../../in/content.csv"
out_folder = "../../out/experiments/bert"

MIN_CONTENT = 15
MIN_WORDS = 30

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

SEED_VALUE = 22

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed_all(SEED_VALUE)

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




def run_experiment(data_train,data_val,data_test, name, bert_variant="bert-base-uncased",use_images=False,subreddits_to_remove=None, version=1):
    start = datetime.datetime.now()

    train = data_train.copy()
    
    if subreddits_to_remove:
        #eliminar subreddits origen
        train = train[~train["first_found_in"].isin(subreddits_to_remove)]


    dataset = DatasetDict({
        "train": Dataset.from_pandas(train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(data_val.reset_index(drop=True)),
        "test": Dataset.from_pandas(data_test.reset_index(drop=True))
    })
    



    tokenizer = AutoTokenizer.from_pretrained(bert_variant)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    labels = ["True","False"]
    id2label = {1: "True", 0: "False"}
    label2id = {"True":1,"False":0}

    def preprocess_data(examples, max_length):
        # take a batch of texts
        texts = examples["text"]
        # encode the
        encoding = tokenizer(
                texts,
                #max_length=max_length,
                #padding="max_length", 
                truncation=True, 
                max_length=256
            )
        
        encoding["labels"] = [int(label) for label in examples["has_ADHD"]]
        encoding["user"] = examples["user"]
        encoding["has_ADHD"] = examples["has_ADHD"]
        if use_images:
            encoding["text"] = [((a or "") + " " + (b or "")).strip() for a, b in zip(examples["text"], examples["image_description"])]
        else:
            encoding["text"] = examples["text"]

        encoding["content"] = examples["id_y"]
            
        return encoding



    if bert_variant == "cambridgeltl/BioRedditBERT-uncased":
        max = 512
    else:
        max = None
        
    encoded_dataset = dataset.map(lambda x: preprocess_data(x,max), batched=True, remove_columns=dataset['train'].column_names)
    encoded_dataset.set_format("torch")



    model = AutoModelForSequenceClassification.from_pretrained(
        bert_variant,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )



    batch_size = 1
    metric_name = "f1"

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                tuple) else p.predictions
        
        y_pred = np.argmax(preds,axis=1)
        y_true = p.label_ids
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        
        probs = torch.softmax(torch.tensor(preds), dim=1).numpy()
        proba_positive = probs[:, 1]  #probabilidad true
        
        return {
            'accuracy': accuracy,
            'f1': f1
        }

    args = TrainingArguments(
        f"bert-finetuned-sem_eval-english",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        fp16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=2,
        save_steps=500,
        gradient_checkpointing=True,
        gradient_accumulation_steps=32,
        dataloader_num_workers=0,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        seed=SEED_VALUE,
        data_seed=SEED_VALUE
        #push_to_hub=True,
    )



    outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0), labels=encoded_dataset['train'][0]['labels'].unsqueeze(0))

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation"],
        #tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()



    #validacion
    val_predictions = trainer.predict(encoded_dataset["validation"])
    y_pred_probs_val = torch.nn.functional.softmax(torch.tensor(val_predictions.predictions), dim=-1)

    val_df = encoded_dataset["validation"].to_pandas()
    val_df['pred_proba_true'] = y_pred_probs_val[:, 1].numpy()



    #test
    test_predictions = trainer.predict(encoded_dataset["test"])
    y_pred_probs_test = torch.nn.functional.softmax(torch.tensor(test_predictions.predictions), dim=-1)

    test_df = encoded_dataset["test"].to_pandas()
    test_df['pred_proba_true'] = y_pred_probs_test[:, 1].numpy()
    
    end = datetime.datetime.now()
    print(f"Total runtime of experiment ({name}) is: {end - start}")

    #agrego por usuario
    #agregar por usuario
    def aggregate_users(df):
        user_stats = df.groupby('user').agg(
            mean_proba=('pred_proba_true', 'mean'),
            real_label=('has_ADHD', 'first'),
            id=('user','first')
        ).reset_index()

        return user_stats
    
    user_val = aggregate_users(val_df)
    user_test = aggregate_users(test_df)


    #GUARDAR NIVEL USUARIO
    #guardar predicciones val
    val_predictions_to_save = user_val[['user', 'real_label', 'mean_proba']].copy()
    val_predictions_to_save.rename(columns={
        'real_label': 'true_label',
        'mean_proba': 'predicted_probability'
    }, inplace=True)
    val_predictions_to_save['dataset'] = 'validation'

    #guardar predicciones test
    test_predictions_to_save = user_test[['user', 'real_label', 'mean_proba']].copy()
    test_predictions_to_save.rename(columns={
        'real_label': 'true_label',
        'mean_proba': 'predicted_probability'
    }, inplace=True)
    test_predictions_to_save['dataset'] = 'test'
    
    data_users = pd.concat([val_predictions_to_save, test_predictions_to_save], axis=0)
    data_users["experiment_type"] = "bert"
    data_users["experiment_name"] = name
    data_users["version"] = version
    
    if not os.path.exists(f"{out_folder}/{name}"):
        os.makedirs(f"{out_folder}/{name}")
        
    data_users.to_csv(f"{out_folder}/{name}/users_predictions.csv",index=False)


    
    #GUARDAR NIVEL CONTENIDO
    #predicciones val
    val_predictions_to_save = val_df[['pred_proba_true','has_ADHD']].copy()
    val_predictions_to_save.rename(columns={
        'has_ADHD': 'true_label',
        'pred_proba_true': 'predicted_probability'
    }, inplace=True)
    val_predictions_to_save['dataset'] = 'validation'
    
    #predicciones test
    test_predictions_to_save = test_df[['pred_proba_true','has_ADHD']].copy()
    test_predictions_to_save.rename(columns={
        'has_ADHD': 'true_label',
        'pred_proba_true': 'predicted_probability'
    }, inplace=True)
    test_predictions_to_save['dataset'] = 'test'

    datat_content = pd.concat([val_predictions_to_save, test_predictions_to_save], axis=0)
    datat_content["experiment_type"] = "bert"
    datat_content["experiment_name"] = name
    datat_content["version"] = version
    
    datat_content.to_csv(f"{out_folder}/{name}/content_predictions.csv",index=False)
        
        
        
users_df = pd.read_csv(user_file)
content_df = pd.read_csv(content_file,dtype={"image_description": str})
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


variants={
    "BERT":"bert-base-uncased",
    "RoBERTa":"FacebookAI/roberta-base",
    "DistilBERT":"distilbert/distilbert-base-uncased",
    "ALBERT":"albert/albert-base-v2",
    "BioRedditBERT":"cambridgeltl/BioRedditBERT-uncased",
    "ADHDoBERTa":"../../out/roberta-domain-adapted"
}

for name, variant in variants.items():
    run_experiment(train,val,test,f"{name}_allsources_textonly",variant,False,None,3)
    run_experiment(train,val,test,f"{name}_removedofftopic_textonly",variant,False,["r/self","r/CasualConversation"],3)
    run_experiment(train,val,test,f"{name}_allsources_withdescription",variant,True,None,3)
    run_experiment(train,val,test,f"{name}_removedofftopic_withdescription",variant,True,["r/self","r/CasualConversation"],3)
