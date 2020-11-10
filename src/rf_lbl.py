import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    features = [x for x in df.columns if x not in ["id", "target", "kfold"]]

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    xtrain = df_train.loc[:, features].values
    xvalid = df_valid.loc[:, features].values

    lbl_enc = preprocessing.LabelEncoder()
    ytrain = lbl_enc.fit_transform(df_train.target.values)
    yvalid = lbl_enc.fit_transform(df_valid.target.values)

    rf = ensemble.RandomForestClassifier()
    rf.fit(xtrain, ytrain)
    pred = rf.predict_proba(xvalid)[:, 1]

    log_loss = metrics.log_loss(yvalid, pred)
    print(f"fold={fold}, loss={log_loss}")

    df_valid.loc[:, "rf_lbl_pred"] = pred

    return df_valid[["id", "sentiment", "kfold", "rf_lbl_pred"]]

if __name__ == "__main__":
    dfs = []
    for j in range(5):
        temp_df = run(j)
        dfs.append(temp_df)

    fin_valid_df = pd.concat(dfs)
    print(fin_valid_df.shape)
    fin_valid_df.to_csv("../model_preds/rf_lbl.csv", index=False)
