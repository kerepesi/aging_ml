import pandas as pd
import numpy as np
import sys
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn.cross_validation import KFold


IN_name=sys.argv[4]
X_raw = pd.read_csv(IN_name, dtype=np.float32)
X=X_raw.values
y = np.ravel(pd.read_csv("uniprot_sprot_human.dat-label_col_agingGOs_or_GenAge.csv", usecols=[1], sep=",", dtype=np.float32))

thresholds = np.linspace(0.01, 0.99, 50)
max_d=int(sys.argv[1])
n_est=int(sys.argv[2])
n_exp=int(sys.argv[3])
print("n_est: "+str(n_est))

avg_auc=[]
first=1
rnd_numbers=np.random.randint(1000, size=(n_exp))
for rnd in rnd_numbers:
    print("\nrnd",rnd)
    auc_list=[]
    clf = XGBClassifier(learning_rate=0.3, max_depth=max_d, nthread=20, n_estimators=n_est)
    kf = StratifiedKFold(y,n_folds=5, random_state=rnd, shuffle=True)
    preds = np.ones(y.shape[0])
    i=0
    for train, test in kf:
        i+=1
        preds[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]
        fold_auc=roc_auc_score(y[test], preds[test])
    auc=roc_auc_score(y, preds)
    print("auc",auc)
    avg_auc.append(auc)
    if first==1:
        X_raw["preds"]=preds
        first=0
    else:
        X_raw["preds"]+=preds
print("avg_auc= "+str(np.mean(avg_auc)))
print("std_auc= "+str(np.std(avg_auc)))
OUT_name=IN_name+"-XGBoost_CV_preds-n_est"+str(n_est)+"-exp"+str(n_exp)+".csv"
print(OUT_name)
X_raw[["preds"]].div(len(rnd_numbers)).to_csv(OUT_name,index=False)
