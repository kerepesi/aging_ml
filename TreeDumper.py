import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import sys

def dump_model(model,num_trees=None,output_file=None):
    "Dump the first 'num_trees' tree from XGBClassifier model to file"
    booster = model.booster()
    trees = booster.get_dump()
    if num_trees == None:
        num_trees = len(trees)
        oster = model.booster()
    tree_separator = "booster[%i]:\n"
    model_str = ""
    for i in range(num_trees):
        model_str += tree_separator % i
        model_str += trees[i]
    if output_file != None:
        with open(output_file,'w') as f:
            f.write(model_str)
    else:
        return model_str

IN_name=sys.argv[3]

X_train = pd.read_csv(IN_name, dtype=np.float32)
y_train=np.ravel(pd.read_csv("../DataCreation/uniprot_sprot_human.dat-label_col.csv", usecols=[1], sep=",", dtype=np.float32))

max_d=int(sys.argv[1])
n_est=int(sys.argv[2])

model = XGBClassifier(learning_rate=0.3, max_depth=max_d, nthread=10, n_estimators=n_est)
model.fit(X_train, y_train)

d={}
imp_scores=list(model.feature_importances_)
for i in range(len(imp_scores)):
    d[i]=imp_scores[i]
for k in sorted(d, key=d.get, reverse=True):
    if d[k]>0:
        print(X_train.columns[k],round(d[k],3))

OUT_name=IN_name+"_Trees-n_est"+str(n_est)+"-max_d"+str(max_d)+".txt"
print(OUT_name)
dump_model(model,num_trees=None,output_file=OUT_name)
