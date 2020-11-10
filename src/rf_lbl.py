import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        lbl_enc.fit(df[feat].values)
        lbl_enc.transform(df[feat].values)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    xtrain = df_train.loc[:, features].values
    xvalid = df_valid.loc[:, features].values

    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    rf = ensemble.RandomForestClassifier()
    rf.fit(xtrain, ytrain)
    pred = rf.predict_proba(xvalid)
    print(pred)



if __name__ == "__main__":
    run(5)
