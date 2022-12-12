import pandas as pd
import math
import random
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, auc, \
    average_precision_score
from sklearn.model_selection import train_test_split

positive_grades = [4, 5, 6, 7]
negative_grades = [0, 1, 2, 3]
threshold = 0.5

def info(data):
    sum = 0
    size = len(data.index)
    for val, count in data['GRADE'].value_counts().items():
        quotient = count / size
        sum -= quotient * math.log2(quotient)
    return sum


def infox(all_data, separated_data):
    sum = 0
    all_size = len(all_data.index)
    for _, data in separated_data:
        size = len(data.index)
        sum += (size / all_size) * info(data)
    return sum


def split_info(all_data, separated_data):
    sum = 0
    all_size = len(all_data.index)
    for _, data in separated_data:
        size = len(data.index)
        sum -= (size / all_size) * math.log2(size / all_size)
    return sum


def gain_ratio(all_data, separated_data):
    si = split_info(all_data, separated_data)
    if si != 0:
        return (info(all_data) - infox(all_data, separated_data)) / si
    else:
        return 0.0


def separate_data(data, attr_name):
    return data.groupby(attr_name)


def prob_class(data):
    max_count = 0
    max_class = 0
    size = len(data.index)
    for clazz, count in data['GRADE'].value_counts().items():
        if count > max_count:
            max_count = count
            max_class = clazz
    prob = float(max_count) / float(size)
    return max_class, prob


def find_max_attr(data, attributes):
    max_gr = -1
    max_attr = ''
    for attr in attributes:
        curr_gr = gain_ratio(data, separate_data(data, attr))
        if curr_gr > max_gr:
            max_gr = curr_gr
            max_attr = attr
    return max_attr


class Des_Tree:
    attribute: str
    children: dict#[str, 'Des_Tree']
    predicted_class: int
    prob_class: float

    def __init__(self, attribute=None, children=None, predicted_class=None, prob_class=None):
        if children is None:
            children = dict()
        self.attribute = attribute
        self.children = children
        self.prob_class = prob_class
        self.predicted_class = predicted_class


def make_tree(data, attributes):
    max_attr = ''
    if len(attributes) > 1:
        max_attr = find_max_attr(data, attributes)
    else:
        max_attr = attributes[0]
    separated_data = separate_data(data, max_attr)
    pred, prob = prob_class(data)
    curr = Des_Tree(attribute=max_attr, predicted_class=pred, prob_class=prob)  # сюда разные значения
    if len(attributes) > 1:
        for i in separated_data.groups.keys():
            new_attrs = attributes.copy()
            new_attrs.remove(max_attr)
            curr.children[i] = make_tree(separated_data.get_group(i), new_attrs)
    return curr


def classify(row, tree: Des_Tree) -> tuple:
    while tree.children:
        attr = tree.attribute
        val = row[attr]
        if val not in tree.children:
            break
        tree = tree.children[val]
    return tree.predicted_class, tree.prob_class


def draw_plt(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    auc_roc = auc(fpr, tpr)
    auc_pr = average_precision_score(y_true, y_score)
    print(f'AUC ROC = {auc_roc}')
    print(f'AUC PR = {auc_pr}')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    plt.show()


# Выполнение

data = pd.read_csv('DATA.csv', sep=';')
data['GRADE'] = data['GRADE'].replace(negative_grades, 0)
data['GRADE'] = data['GRADE'].replace(positive_grades, 1)

attributes = list(data.columns[1: 32])
attr_count = int(math.sqrt(len(attributes)))
random.shuffle(attributes)
attributes = attributes[0: attr_count]
train, test = train_test_split(data, test_size=0.2)
print(attributes)

tree = make_tree(train, attributes)

size = len(test.index)
tn = 0
tp = 0
fn = 0
fp = 0
predict = []
expect = []
probs = []
for index, row in test.iterrows():
    real = row["GRADE"]
    predicted_class, prob = classify(row, tree)
    spaces_count = len(str(index))
    predict.append(predicted_class)
    expect.append(real)
    probs.append(prob)
    if real == predicted_class:
        if real:
            tp += 1
        else:
            tn += 1
    else:
        if real:
            fn += 1
        else:
            fp += 1
    print(f'{index}{" " * (3 - spaces_count)} : {real} : {predicted_class} : {"%.3f" % prob} : {real == predicted_class}')

accuracy = (tp + tn) / size
precision = tp / (tp + fp)
recall = tp / (tp + fn)

print(f'accuracy = {accuracy}')
print(f'precision = {precision}')
print(f'recall = {recall}')
for i in range(len(probs)):
    if predict[i] == 0:
        probs[i] = 1 - probs[i]
draw_plt(expect, probs)
