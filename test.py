import pickle
from utils.utils import normalize_text


def test(sentence, model):
    sentence_copy = normalize_text(sentence)
    y_pred = model.predict([sentence_copy])
    return y_pred


if __name__ == '__main__':
    model = pickle.load(open('model/SVC.pkl', 'rb'))
    string = 'Rất tệ, sản phẩm bốc mùi kinh khủng'
    y_pred = test(string, model)
    print('Positive' if y_pred[0] == 0 else 'Negative')
