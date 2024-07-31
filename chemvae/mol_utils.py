import numpy as np
import pandas as pd


def pad_smile(string, max_len, padding='right'):
    if len(string) <= max_len:
        if padding == 'right':
            return string + " " * (max_len - len(string))
        elif padding == 'left':
            return " " * (max_len - len(string)) + string
        elif padding == 'none':
            return string


def load_smiles_and_data_df(data_file, max_len, reg_tasks=None, logit_tasks=None, normalize_out=None, dtype='float64'):
    # reg_tasks : list of columns in df that correspond to regression tasks
    # logit_tasks : list of columns in df that correspond to logit tasks
    if logit_tasks is None:
        logit_tasks = []
    if reg_tasks is None:
        reg_tasks = []
    df = pd.read_csv(data_file)
    df.iloc[:, 0] = df.iloc[:, 0].str.strip()
    df = df[df.iloc[:, 0].str.len() <= max_len]
    smiles = df.iloc[:, 0].tolist()

    reg_data_df = df[reg_tasks]
    logit_data_df = df[logit_tasks]
    # Load regression tasks
    if len(reg_tasks) != 0 and normalize_out is not None:
        df_norm = pd.DataFrame(reg_data_df.mean(axis=0), columns=['mean'])
        df_norm['std'] = reg_data_df.std(axis=0)
        reg_data_df = (reg_data_df - df_norm['mean']) / df_norm['std']
        df_norm.to_csv(normalize_out)

    if len(logit_tasks) != 0 and len(reg_tasks) != 0:
        return smiles, np.vstack(reg_data_df.values).astype(dtype), np.vstack(logit_data_df.values).astype(dtype)
    elif len(reg_tasks) != 0:
        return smiles, np.vstack(reg_data_df.values).astype(dtype)
    elif len(logit_tasks) != 0:
        return smiles, np.vstack(logit_data_df.values).astype(dtype)
    else:
        return smiles


def smiles_to_hot(smiles, max_len, padding, char_indices, nchars):
    smiles = [pad_smile(i, max_len, padding)
              for i in smiles if pad_smile(i, max_len, padding)]

    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)

    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", smile)
                raise e
    return X
