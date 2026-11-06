'''
    BoW experiments.
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


import os
import datetime




user_file = "../../in/users.csv"
content_file = "../../in/content.csv"
out_folder = "../../out/experiments/bow"

MIN_CONTENT = 15
MIN_WORDS = 30

SEED_VALUE = 22


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

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)



def run_experiment(data_train,data_val,data_test, name, clss_type="nb", use_images=False, subreddits_to_remove=None, version=1):
    start = datetime.datetime.now()


    train = data_train.copy()
    val = data_val.copy()
    test = data_test.copy()
    
    if subreddits_to_remove:
        train = train[~train["first_found_in"].isin(subreddits_to_remove)]


    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True
    )

    if use_images:
        train["text"] = train["text"].astype(str).fillna("") + " " + train["image_description"].astype(str).fillna("")
        val["text"] = val["text"].astype(str).fillna("") + " " + val["image_description"].astype(str).fillna("")
        test["text"] = test["text"].astype(str).fillna("") + " " + test["image_description"].astype(str).fillna("")

    X_train = vectorizer.fit_transform(train["text"])
    X_val = vectorizer.transform(val["text"])
    X_test = vectorizer.transform(test["text"])

    y_train = train["has_ADHD"].astype(int)

    if clss_type == "nb":
        classifier = MultinomialNB(alpha=1.0)
    elif clss_type == "svc":
        svc = LinearSVC(max_iter=10000,C=1.0,random_state=SEED_VALUE)
        classifier = CalibratedClassifierCV(svc,method="sigmoid",cv=5)
    elif clss_type == "lr":
        classifier = LogisticRegression(max_iter=10000,C=1.0,random_state=SEED_VALUE)
    elif clss_type == "sgd":
        classifier = SGDClassifier(loss="log_loss",max_iter=10000,random_state=SEED_VALUE)
    else:
        classifier = RandomForestClassifier(random_state=SEED_VALUE)

    classifier.fit(X_train,y_train)

    val_probs = classifier.predict_proba(X_val)[:,1]
    test_probs = classifier.predict_proba(X_test)[:,1]
    
    end = datetime.datetime.now()

    print(f"Total runtime of experiment ({name}) is: {end - start}")
    
    
    #metricas a nivel de usuario
    val_df = val.copy()
    #val_df['pred_label'] = val_preds
    val_df['pred_proba_true'] = val_probs
    
    test_df = test.copy()
    #test_df['pred_label'] = test_preds
    test_df['pred_proba_true'] = test_probs
    
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
    data_users["experiment_type"] = "bow"
    data_users["experiment_name"] = name
    data_users["version"] = version
    
    if not os.path.exists(f"{out_folder}/{name}"):
        os.makedirs(f"{out_folder}/{name}")
        
    data_users.to_csv(f"{out_folder}/{name}/users_predictions.csv",index=False)


    
    #GUARDAR NIVEL CONTENIDO
    #predicciones val
    val_predictions_to_save = val_df[['id_y', 'pred_proba_true','has_ADHD']].copy()
    val_predictions_to_save.rename(columns={
        'id_y' : 'content',
        'has_ADHD': 'true_label',
        'pred_proba_true': 'predicted_probability'
    }, inplace=True)
    val_predictions_to_save['dataset'] = 'validation'
    
    #predicciones test
    test_predictions_to_save = test_df[['id_y', 'pred_proba_true','has_ADHD']].copy()
    test_predictions_to_save.rename(columns={
        'id_y' : 'content',
        'has_ADHD': 'true_label',
        'pred_proba_true': 'predicted_probability'
    }, inplace=True)
    test_predictions_to_save['dataset'] = 'test'

    datat_content = pd.concat([val_predictions_to_save, test_predictions_to_save], axis=0)
    datat_content["experiment_type"] = "bow"
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
no_adhd_text = join_data.drop(join_data[join_data["has_ADHD_pattern"] == True].index)#[:1000]

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

data_train = no_adhd_text[no_adhd_text['user'].isin(train_users['user'])]
data_val = no_adhd_text[no_adhd_text['user'].isin(val_users['user'])]
data_test = no_adhd_text[no_adhd_text['user'].isin(test_users['user'])]
 


classifiers = [
     "nb",
     "svc",
     "lr",
     "sgd",
     "rf"
]

for c in classifiers:
    run_experiment(data_train,data_val,data_test, f"{c}_allsources_textonly",clss_type=c,use_images=False,subreddits_to_remove=None,version=3)
    run_experiment(data_train,data_val,data_test, f"{c}_removedofftopic_textonly",clss_type=c,use_images=False,subreddits_to_remove=["r/self","r/CasualConversation"],version=3)
    run_experiment(data_train,data_val,data_test, f"{c}_allsources_withdescription",clss_type=c,use_images=True,subreddits_to_remove=None,version=4)
    run_experiment(data_train,data_val,data_test, f"{c}_removedofftopic_withdescription",clss_type=c,use_images=True,subreddits_to_remove=["r/self","r/CasualConversation"],version=4)
