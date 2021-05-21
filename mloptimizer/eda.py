import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

DATA_MASTER_FOLDER = "../../data"


def read_dataset(file_name="train_test.csv", ohe=0, scaler=None, samples=None, return_x_y=False,
                 vars_type=['numerical', 'factor'], sampling=-1, replace=False):
    data_path = os.path.join(DATA_MASTER_FOLDER, file_name)

    df_raw = pd.read_csv(data_path, sep=",")

    if samples is not None:
        df_raw = df_raw.sample(samples)

    # Data have 36 vars, we will use 9
    if 'numerical' in vars_type:
        numerical_columns = ["revenue", "dti_n", "loan_amnt", "fico_n"]
    else:
        numerical_columns = []
    #factor_columns = ["emp_length", "experiencia_c", "purpose", "home_ownership", "addr_state"]
    if 'factor' in vars_type:
        factor_columns = ["purpose", "home_ownership", "addr_state"]
    else:
        factor_columns = []
    target_column = ["Default"]
    useful_columns = list(set(numerical_columns + factor_columns + target_column))

    df = df_raw[useful_columns].copy()
    if sampling > 0:
        df = df.sample(sampling, replace=replace)
    #df.loc[df["Default"] == "Fully Paid", "Default"] = 0
    #df.loc[df["Default"] == "Charged Off", "Default"] = 1
    #df.replace({'Fully Paid': 0, 'Charged Off': 1})
    df["Default"] = df["Default"].apply(lambda x: 1 if x == 0 else 0)
    #df["Default"] = LabelEncoder().fit_transform(df["Default"])

    if scaler is not None:
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    if ohe:
        df_ohe, enc = one_hot_encode(df[factor_columns])
        df.drop(columns=factor_columns, inplace=True)
        df = pd.merge(df, df_ohe, left_index=True, right_index=True)
        return df, enc
    if return_x_y:
        return np.array(df[numerical_columns]).copy(), np.array(df[target_column]).reshape(-1).copy()
    return df


def read_dataset_valida(file_name="validaf.csv", enc=None):
    data_path = os.path.join(DATA_MASTER_FOLDER, file_name)

    df_raw = pd.read_csv(data_path, sep=",")
    df_raw['Default'] = df_raw['loan_status']

    # Data have 30 vars, we will use 9
    numerical_columns = ["revenue", "dti_n", "loan_amnt", "fico_n"]
    #factor_columns = ["emp_length", "experiencia_c", "purpose", "home_ownership", "addr_state"]
    factor_columns = ["purpose", "home_ownership", "addr_state"]
    target_column = ["Default"]
    useful_columns = list(set(numerical_columns + factor_columns + target_column))

    df = df_raw[useful_columns].copy()
    if enc is not None:
        df_ohe, enc = one_hot_encode(df[factor_columns], enc)
        df.drop(columns=factor_columns, inplace=True)
        df = pd.merge(df, df_ohe, left_index=True, right_index=True)
        return df, enc

    return df


def one_hot_encode(df, enc=None):
    new_df = df.copy()
    old_columns = list(df.columns)

    if enc is None:
        enc = OneHotEncoder()
        enc.fit(df)

    new_columns = enc.get_feature_names()
    data_coded = enc.transform(df).toarray().T

    for i in range(0, len(new_columns)):
        new_df[new_columns[i]] = data_coded[i]

    new_df.drop(columns=old_columns, inplace=True)
    new_df = new_df.astype(dtype='int32')

    return new_df, enc
