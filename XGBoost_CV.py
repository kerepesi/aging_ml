import pandas as pd
import numpy as np
import sys
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from sklearn import tree
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score
import sys


IN_name=sys.argv[4]
X_raw = pd.read_csv(IN_name, dtype=np.float32)
X=X_raw.values
y = np.ravel(pd.read_csv("aging_labels.csv", usecols=[1], sep=",", dtype=np.float32))

thresholds = np.linspace(0.01, 0.99, 50)
max_d=int(sys.argv[1])
n_est=int(sys.argv[2])
n_exp=int(sys.argv[3])
print("n_est: "+str(n_est))

avg_all_auc=[]
first=1
rnd_numbers=np.random.randint(1000, size=(n_exp))
for rnd in rnd_numbers:
    print("-------------")
    print("rnd",rnd)
    for par2 in [0]:
        auc_list=[]
        clf = XGBClassifier(learning_rate=0.3, max_depth=max_d, nthread=10, n_estimators=n_est)
        kf = StratifiedKFold(y,n_folds=5, random_state=rnd, shuffle=True)
        preds = np.ones(y.shape[0])
        i=0
        for train, test in kf:
            i+=1
            preds[test] = clf.fit(X[train], y[train]).predict_proba(X[test])[:,1]
            auc=roc_auc_score(y[test], preds[test])
            auc_list.append(auc)
            print("fold {}, ROC AUC: {:.3f}".format(i, auc))
        mean_auc=np.mean(np.array(auc_list))
        all_auc=roc_auc_score(y, preds)
        mcc = np.array([matthews_corrcoef(y, preds>thr) for thr in thresholds])
        best_threshold = thresholds[mcc.argmax()]
        max_mcc = mcc.max()
        print("mcc.max="+str(max_mcc))
        print("best_threshold="+str(best_threshold))
        print("all_auc= "+str(all_auc))
        avg_all_auc.append(all_auc)
        fp, tp, tn, fn = 0,0,0,0
        for i in range(len(preds)):
            if preds[i] > best_threshold and y[i]==0:
                fp+=1
            elif preds[i] > best_threshold and y[i]==1:
                tp+=1
            elif preds[i] <= best_threshold and y[i]==0:
                tn+=1
            elif preds[i] <= best_threshold and y[i]==1:
                fn+=1
    print("fp, tp, tn, fn")
    print(str(fp)+","+str(tp)+","+str(tn)+","+str(fn))
    print(str(tp+fn)+","+str(fp+tp+tn+fn))
    if first==1:
        X_raw["preds"]=preds
        first=0
    else:
        X_raw["preds"]+=preds
print("avg_all_auc= "+str(np.mean(avg_all_auc)))
print("std_all_auc= "+str(np.std(avg_all_auc)))
OUT_name=IN_name+"-XGBoost_CV_preds-n_est"+str(n_est)+"-exp"+str(n_exp)+".csv"
print(OUT_name)
X_raw[["preds"]].div(len(rnd_numbers)).to_csv(OUT_name,index=False)
