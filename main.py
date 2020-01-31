import pandas as pd
from utils.utils import normalize_text, no_marks


def preprocee_data(file_path):
    df = pd.read_csv(file_path)
    df_augmentation = df.copy()
    df['text'].apply(lambda x: normalize_text(x))
    df_augmentation['text'](lambda x: normalize_text(x))
    df_augmentation['text'](lambda x: no_marks(x))

    df.append(df_augmentation)

    return df

if __name__ == '__main__':

    train_path = 'data/processed/train.csv'
    preprocee_data(train_path)