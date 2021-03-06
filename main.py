import pandas as pd
from utils.utils import normalize_text, no_marks
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

stop_ws = (u'rằng', u'thì', u'là', u'mà')


def preprocee_data(file_path):
    df = pd.read_csv(file_path)
    df_augmentation = df.copy()
    df['clean_text'] = df['text'].apply(lambda x: normalize_text(x))
    df_augmentation['clean_text'] = df_augmentation['text'].apply(lambda x: normalize_text(x))
    df_augmentation['clean_text'] = df_augmentation['text'].apply(lambda x: no_marks(x))

    df.append(df_augmentation)

    return df


def train(file_path, list_classifiers):
    data_clean = preprocee_data(file_path)

    X_train, X_test, y_train, y_test = train_test_split(data_clean['clean_text'].values.tolist(), data_clean['label'].values.tolist(), random_state=10)

    print(len(X_train),len(X_test))

    print(X_train[:3])

    for cls in list_classifiers:
        steps = []

        steps.append(('CountVectorizer', CountVectorizer(ngram_range=(1, 5), stop_words=stop_ws, max_df=0.5, min_df=5)))
        steps.append(('Tfidf', TfidfTransformer(use_idf=False, sublinear_tf=True, norm='l2', smooth_idf=True)))
        steps.append(('classifier', cls))

        clf = Pipeline(steps=steps)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, labels=[1, 0], digits=3)

        model_name = type(cls).__name__

        with open(model_name,'wb') as f:

            pickle.dump(clf,f)

        print(report)


if __name__ == '__main__':
    train_path = '/content/VN-Sentiment/data/processed/train.csv'
    list_classifiers = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        SVC(C=1),
        LogisticRegression()
    ]
    train(train_path,list_classifiers)