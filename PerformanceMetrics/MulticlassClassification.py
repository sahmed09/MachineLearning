from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

C = 'Cat'
D = 'Dog'
F = 'Fox'

# True Values
y_true = [C, C, C, C, C, C, F, F, F, F, F, F, F, F, F, F, D, D, D, D, D, D, D, D, D]

# Predicted Values
y_pred = [C, C, C, C, D, F, C, C, C, C, C, C, D, D, F, F, C, C, C, D, D, D, D, D, D]

# The precision for the Cat class is the number of correctly predicted Cat out of all predicted Cat
# The recall for the Cat class is the number of correctly predicted Cat out of actual Cat

# Print the confusion matrix, precision, recall, classification report
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print('Accuracy:', accuracy_score(y_true, y_pred))
print('Precision:', precision_score(y_true, y_pred, average='weighted'))
print('Recall:', recall_score(y_true, y_pred, average='weighted'))
print('F1 Score:', f1_score(y_true, y_pred, average='weighted'))
