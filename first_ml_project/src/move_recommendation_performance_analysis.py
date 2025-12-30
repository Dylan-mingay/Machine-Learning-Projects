# Import libraries for performance metrics and plotting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


def analyze_performance(Y_test, prediction):
    # Compute the confusion matrix for binary classification
    cm = confusion_matrix(Y_test, prediction, labels=[0, 1])
    # Calculate precision: TP / (TP + FP)
    precision = precision_score(Y_test, prediction, pos_label=1)
    # Calculate recall: TP / (TP + FN)
    recall = recall_score(Y_test, prediction, pos_label=1)
    # Calculate F1-score: 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(Y_test, prediction, pos_label=1)

    # Print a detailed classification report including precision, recall, F1, and support
    print(classification_report(Y_test, prediction))

    return cm, precision, recall, f1

def receiver_operating_characteristic(Y_test, prediction_probabilities):
    # Extract probabilities for the positive class (class 1)
    pos_probs = prediction_probabilities[:, 1]
    # Define thresholds from 0.0 to 1.0 in steps of 0.05
    thresholds = np.arange(0.0, 1.1, 0.05)
    
    # Initialize lists to count true positives and false positives at each threshold
    true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    # For each prediction and true label, count TPs and FPs across thresholds
    for pred, y in zip(pos_probs, Y_test):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                if y == 1:
                    true_pos[i] += 1  # True positive
                else:
                    false_pos[i] += 1  # False positive
            else:
                break  # No need to check lower thresholds for this prediction

    # Count total positive and negative samples in the test set
    n_pos = (Y_test==1).sum()
    n_neg = (Y_test==0).sum()
    # Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for each threshold
    tpr = [tp/n_pos for tp in true_pos]
    fpr = [fp/n_neg for fp in false_pos]

    # Create a new figure for the ROC plot
    plt.figure()
    lw = 2
    # Plot the ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=lw)
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # Set axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Label axes and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # Display the plot
    plt.show()

    # Print the Area Under the Curve (AUC) score
    print(roc_auc_score(Y_test, pos_probs))