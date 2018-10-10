from flask import Flask, Blueprint, jsonify
import socket
import time
import joblib
import sklearn
import json
import flask
import scipy
import pprint
import redis
import pickle

import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold
import sklearn.preprocessing as pp
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


import datetime
import pytz
import uuid

import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def now_localtz():
    return datetime.datetime.now(pytz.timezone('Europe/Lisbon'))

VERSION="20181005-8"
DATE_STARTED=now_localtz()
HOSTNAME=joblib.hash("salted2662"+socket.gethostname())
WORKER_ID=str(uuid.uuid4())

USE_CACHE=False
REDIS_HOST="XXXXXXXXX.redis.cache.windows.net"
REDIS_KEY="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
CACHE_VERSION="v3"

app = Flask(__name__)


# LogicApps configured to send all Msft Forms full form body with question Ids and forms answers
# We map here each question id to sklearn algorithm and pipeline parameters

FORM_IDS={
    "rf71efaaee75f4869b3a24de441b09919":"algorithm",
    "r52e336e1f3564f47b9359debc320a7ce":"nickname",
    "r71b640ccadb844af885b17eb733c4a8b":"logreg__penalty",
    "r462aa2316f0f4819a7bf1e20bd729975":"logreg__C",

    "rd24bc7e764d34b1b83c8d3acf2a91203":"rf__n_estimators",
    "re36df3b7827a41b38742ebab3a9d09d5":"rf__criterion",
    "r189c2542d50b491f8086f02963b7a081":"rf__max_depth",
    "rc9525cd279d14be38771f6f40e4316e4":"rf__min_samples_leaf",
    "rce0c88b0fb9248498ff46de546904f63":"rf__max_features",

    "r6ea09cddd77447e7b5391b31f7945537":"dt__criterion",
    "rba34bf1b855d4ec9a29a74585dcb6bae":"dt__max_depth",
    "rdf6bfb2bd41b405b9dc170e39a6d5154":"dt__min_samples_split",
    "r674ff05de69a4cacbf1e505e55c76281":"dt__min_samples_leaf",
    "r56862ced92a64fd598fbeabc0cbc8d67":"dt__max_features",

    "r2153f109fe2b418795a01c53b500af04":"svm__kernel",
    "rb62cd150e455454c8098824e05a4c8b5":"svm__degree",
    "r96df4ff1491246e4a6d3ad95148f61cb":"svm__C",
    
    "reea67b283e9d446096c7c3ab825169bf":"xt__n_estimators",
    
    "r4849ae4db1c34253b08dac5b9a66de63":"pca",
    
    "r4ca8638f21674388b85b1ffe385a8742":"text_preproc"
}

@app.route("/")
def hello():
    hostname=socket.gethostname()
    return f"App Version: {VERSION}\r\nHostname:{HOSTNAME}\r\nWorker Id:{WORKER_ID}\r\nDate Started:{DATE_STARTED}".replace("\r\n","<br>")



@app.route("/train",methods=["POST"])
def train_route():
    
    time_started = now_localtz()
    
    results={}
    
    try:
        print(flask.request.data)
        raw_params=json.loads(flask.request.data)
        
        # Translate form question keys ids into friendly keys
        form_params={}
        for k in raw_params.keys():
            if FORM_IDS.get(k):
                form_params[FORM_IDS[k]]=raw_params[k]
        
        print(form_params)
        
        # Call inner train
        scores=train(**form_params)        
        
        results["status"]="ok"

        results["scores"]=','.join(str(x) for x in scores)        
        results["score_mean"]=np.mean(scores)
        results["score_std"]=np.std(scores)
        results["score_hmean"]=scipy.stats.hmean(scores)
        
    except Exception as error:
        results["exception"]=str(error)
        results["status"]="error"

        # Numbers cannot be null or "" :(
        results["score_std"]=0
        results["score_hmean"]=0
        results["score_mean"]=0
        
        print("Error: %s"%(error))
        pass
    finally:
        time_ended = now_localtz()
    
    
    results["notes"]=str(form_params)
    results["timestamp"]=time_ended.isoformat()
    
    # Add Hour/time/Second to each submission nickname
    nickname=form_params.get("nickname","-")[0:10]+" ("+now_localtz().strftime("%H_%M_%S")+")"    
    results["nickname"]=nickname
    results["duration_secs"]=(time_ended - time_started).total_seconds()
    
    results["algorithm"]=form_params.get("algorithm","-")
    results["app_version"]=VERSION
    results["host"]=HOSTNAME
    print(results)
    return (jsonify(results))
        

class DataFrameImputer(TransformerMixin):

    def __init__(self, default_value="NA"):
        self.default_value = default_value
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame(X).fillna(self.default_value)

# Dataset loads here
df_train=pd.read_csv("train.csv")
y=df_train.pop("Survived")
X=df_train

# Cache (redis)
if USE_CACHE:
    cache = redis.StrictRedis(host=f'{REDIS_HOST}',
            port=6380, db=0, password=f'{REDIS_KEY}', ssl=True)

    cache_ping=cache.ping()

    print("Redis Ping returned : " + str(cache_ping))
else:
    cache=None

# Main train function
def train(**kargs):
      
    random_state=43
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=random_state)
    
    if kargs["algorithm"]=="Logistic Regression":
        clf=LogisticRegression(random_state=random_state)
        clf_name="logreg"
    
    if kargs["algorithm"]=="Random Forest":
        clf=RandomForestClassifier(random_state=random_state)
        clf_name="rf"
    
    if kargs["algorithm"]=="Decision Tree":
        clf=DecisionTreeClassifier(random_state=random_state)
        clf_name="dt"
    
    if kargs["algorithm"]=="SVM":
        clf=SVC(random_state=random_state)
        clf_name="svm"
    
    if kargs["algorithm"]=="Extra Trees":
        clf=ExtraTreesClassifier(random_state=random_state)
        clf_name="xt"
        
    print("train params",kargs)
    
    pipeline=[]
    
    # Basic post prep pipeline (onehot/remove any remaining NA), make the dataset scikit compliant
    nums=[ ([c],pp.Imputer()) for c in X.select_dtypes(np.number)]
    cats=[ ([c],[DataFrameImputer(default_value=""), pp.LabelBinarizer()]) for c in X.select_dtypes("object")]
    
    texts=[]
    text_preproc=kargs.get("text_preproc")
    if text_preproc and text_preproc!="None":
        if text_preproc=="Tfidf":
            texts=[ ("Name",TfidfVectorizer())]
        elif text_preproc=="Count":
            texts=[ ("Name",CountVectorizer())]
        else:
            raise(Exception(f"not valid:{text_preproc}"))
    
    print(texts)
    mapper=DataFrameMapper(nums+cats+texts,df_out=True)

    pipeline.append(('featurize', mapper))
        
    pca=kargs.get("pca")
    if pca and pca!="Disabled":
        print("add pca")
        pipeline.append(('pca', PCA(n_components=guess_type(kargs["pca"]))))
        
    pipeline.append((clf_name,clf))
    
     # Our full pipeline
    train_pipeline=Pipeline(pipeline)
    
    # Set classifier parameters
    for k in kargs.keys():
        if (clf_name+"__") in k:            
            train_pipeline.set_params(**{k:guess_type(kargs[k])})
    # Dump
    for step in train_pipeline.steps:
        pprint.pprint(step)
        
    # Check cache
    if USE_CACHE:
        cache_key=CACHE_VERSION+"__"+str(joblib.hash(train_pipeline))
        print("Cache key:",cache_key)
        scores=cache.get(cache_key)        
        print("From Cache")
        scores=pickle.loads(scores)
        return scores+np.random.normal(0,.0005,len(scores))*100
    
    print("Not in cache, training...")
    
    # Train/Cross eval
    scores=cross_val_score(X=X,y=y,cv=rskf,estimator=train_pipeline,verbose=5,n_jobs=1,scoring="accuracy")
    scores=(scores*100).round(3)
   
    if USE_CACHE:
        print("Saving in cache...")
        cache.set(cache_key,pickle.dumps(scores))
    
    return scores+np.random.normal(0,.0005,len(scores))*100


def guess_type(s):
    if not isinstance(s,str):
        return s
    if s=="" or s=="None" or s=="none":
        return None
    try:
        if np.isclose(float(s),int(s)):
            return (int(s))
    except:
        try:
            return (float(s))
        except:
            try:
                return (int(s))
            except:
                return str(s)
                pass
            pass
        pass