import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


# Step 1: Create and save "normal" and "abnormal" arrays to binary files
def create_data():
    # Create "normal" array with random values between 0 and 10
    normal = np.random.uniform(0, 10, size=(100, 1000))
    normal.tofile("normal.bin")
    # print(normal.shape)  # shape represents (number of rows, number of columns)

    # Create "abnormal" array with random values between 5 and 15
    abnormal = np.random.uniform(5, 15, size=(100, 1000))
    abnormal.tofile("abnormal.bin")


# Step 2: Load the saved data into two arrays
def load_data():
    normal = np.fromfile("normal.bin").reshape(100, 1000)
    abnormal = np.fromfile("abnormal.bin").reshape(100, 1000)

    # print(normal)
    # print(abnormal)
    return normal, abnormal


# Step 3: Split data into training and test sets
def split_data(normal, abnormal):
    training, normal_test = normal[:90], normal[90:]
    abnormal_test = abnormal[:10]

    test = np.vstack((normal_test, abnormal_test))
    # test2 = np.concatenate((normal_test, abnormal_test), axis=0)
    # test_labels = np.array([0] * len(normal_test) + [1] * len(abnormal_test))

    # return training, test, test_labels
    return training, test


# Step 4: Calculate dissimilarity scores for the training set
def calculate_dissimilarity_scores(training):
    baseline = []
    for i in range(len(training)):
        distances = euclidean_distances(training[i].reshape(1, -1), training)
        # top_distances = np.partition(distances, 6)[:, 1:6]  # Exclude self-distance
        sorted_distances = np.sort(distances)
        top_distances = sorted_distances[:, 1:6]
        score = top_distances.sum()
        baseline.append(score)
    return baseline


# Step 5: Detect anomalies in the test set
def print_predictions(training, test, baseline):
    test_scores = []
    for i in range(len(test)):
        distances = euclidean_distances(test[i].reshape(1, -1), training)
        # top_distances = np.partition(distances, 6)[:, 1:6]  # Exclude self-distance
        sorted_distances = np.sort(distances)
        top_distances = sorted_distances[:, 1:6]
        score = top_distances.sum()
        test_scores.append(score)

    min_baseline_value = min(baseline)
    max_baseline_value = max(baseline)

    # print(min_baseline_value)
    # print(max_baseline_value)
    # print(min(test_scores))
    # print(max(test_scores))
    # print(test_scores)

    predictions = []
    for score in test_scores:
        if min_baseline_value <= score <= max_baseline_value:
            predictions.append("Normal")
        else:
            predictions.append("Abnormal")

    return predictions


if __name__ == "__main__":
    create_data()
    normal, abnormal = load_data()
    training, test = split_data(normal, abnormal)
    baseline = calculate_dissimilarity_scores(training)
    predictions = print_predictions(training, test, baseline)

    for i, prediction in enumerate(predictions):
        print(f"Data {i}: {prediction}")


"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


# Step 1: Create and save "normal" and "abnormal" arrays to binary files
def create_and_save_data():
    # Create "normal" array with random values between 0 and 10
    normal_array = np.random.uniform(0, 10, size=(100, 1000))
    normal_array.tofile("normal.bin")

    # Create "abnormal" array with random values between 5 and 15
    abnormal_array = np.random.uniform(5, 15, size=(100, 1000))
    abnormal_array.tofile("abnormal.bin")


# Step 2: Load the saved data into two arrays
def load_data():
    normal_data = np.fromfile("normal.bin").reshape(100, 1000)
    abnormal_data = np.fromfile("abnormal.bin").reshape(100, 1000)
    return normal_data, abnormal_data


# Step 3: Split data into training and test sets
def split_data(normal_data, abnormal_data):
    normal_train, normal_test = normal_data[:90], normal_data[90:]
    abnormal_test = abnormal_data[:10]

    test_data = np.vstack((normal_test, abnormal_test))
    test_labels = np.array([0] * len(normal_test) + [1] * len(abnormal_test))

    return normal_train, test_data, test_labels


# Step 4: Calculate dissimilarity scores for the training set
def calculate_baseline_scores(normal_train):
    baseline_scores = []
    for i in range(len(normal_train)):
        distances = euclidean_distances(normal_train[i].reshape(1, -1), normal_train)
        top_distances = np.partition(distances, 6)[:, 1:6]  # Exclude self-distance
        score = top_distances.sum()
        baseline_scores.append(score)
    return baseline_scores


# Step 5: Detect anomalies in the test set
def detect_anomalies(test_data, normal_train, baseline_scores):
    scaler = StandardScaler()
    baseline_scores = np.array(baseline_scores)
    baseline_scores = scaler.fit_transform(baseline_scores.reshape(-1, 1))

    test_scores = []
    for i in range(len(test_data)):
        distances = euclidean_distances(test_data[i].reshape(1, -1), normal_train)
        top_distances = np.partition(distances, 6)[:, 1:6]  # Exclude self-distance
        score = top_distances.sum()
        test_scores.append(score)

    test_scores = np.array(test_scores)
    test_scores = scaler.transform(test_scores.reshape(-1, 1))

    min_score = min(baseline_scores)
    max_score = max(baseline_scores)

    predictions = []
    for score in test_scores:
        if min_score <= score <= max_score:
            predictions.append("Normal")
        else:
            predictions.append("Abnormal")

    return predictions


if __name__ == "__main__":
    create_and_save_data()
    normal_data, abnormal_data = load_data()
    normal_train, test_data, test_labels = split_data(normal_data, abnormal_data)
    baseline_scores = calculate_baseline_scores(normal_train)
    predictions = detect_anomalies(test_data, normal_train, baseline_scores)

    for i, prediction in enumerate(predictions):
        print(f"Data point {i}: {prediction}")
"""
