from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


def analyze_performance(Y_test, prediction):
    cm = confusion_matrix(Y_test, prediction, labels=[0, 1])
    precision = precision_score(Y_test, prediction, pos_label=1)
    recall = recall_score(Y_test, prediction, pos_label=1)
    f1 = f1_score(Y_test, prediction, pos_label=1)

    # Print detailed classification report
    print(classification_report(Y_test, prediction))

    return cm, precision, recall, f1

def receiver_operating_characteristic(Y_test, prediction_probabilities):
    pos_probs = prediction_probabilities[:, 1]
    thresholds = np.arange(0.0, 1.1, 0.05)
    
    true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    for pred, y in zip(pos_probs, Y_test):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                if y == 1:
                    true_pos[i] += 1
                else:
                    false_pos[i] += 1

            else:
                break

    n_pos = (Y_test==1).sum()
    n_neg = (Y_test==0).sum()
    tpr = [tp/n_pos for tp in true_pos]
    fpr = [fp/n_neg for fp in false_pos]

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print(roc_auc_score(Y_test, pos_probs))