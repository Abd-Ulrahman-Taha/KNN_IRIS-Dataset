import csv
import math
import random
import matplotlib.pyplot as plt

def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            dataset.append(row)
    return dataset

def preprocess_dataset(dataset):
    processed_dataset = []
    class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}  # Mapping for class labels
    
    for data in dataset:
        processed_data = [float(value) for value in data[:-1]]
        processed_data.append(class_mapping[data[-1]])
        processed_dataset.append(processed_data)
        
    return processed_dataset

def euclidean_distance(point1, point2):
    squared_distance = 0
    for i in range(len(point1)):
        squared_distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(squared_distance)

def get_neighbors(train_data, test_instance, k):
    distances = []
    for train_instance in train_data:
        distance = euclidean_distance(train_instance[:-1], test_instance[:-1])
        distances.append((train_instance, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = [item[0] for item in distances[:k]]
    return neighbors

def predict_class(neighbors):
    class_votes = {}
    for neighbor in neighbors:
        class_label = neighbor[-1]
        if class_label in class_votes:
            class_votes[class_label] += 1
        else:
            class_votes[class_label] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

def k_nearest_neighbors(train_data, test_data, k):
    predictions = []
    for test_instance in test_data:
        neighbors = get_neighbors(train_data, test_instance, k)
        predicted_class = predict_class(neighbors)
        predictions.append(predicted_class)
    return predictions

def evaluate_model(predictions, true_labels):
    correct_predictions = sum(1 for i in range(len(predictions)) if predictions[i] == true_labels[i])
    accuracy = correct_predictions / len(predictions)
    return accuracy

def plot_data(dataset):
    classes = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    colors = ['red', 'green', 'blue']

    for data in dataset:
        features = data[:-1]
        label = data[-1]
        class_name = classes[label]
        plt.scatter(features[0], features[1], color=colors[label], label=class_name)

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Iris Flower Dataset')
    plt.legend()
    plt.show()

def predict_class_label(train_data, input_data, k):
    distances = []
    for train_instance in train_data:
        distance = euclidean_distance(train_instance[:-1], input_data)
        distances.append((train_instance, distance))
    distances.sort(key=lambda x: x[1])
    neighbors = [item[0] for item in distances[:k]]
    predicted_class = predict_class(neighbors)
    return predicted_class

# Set the hyperparameters
k = int(input("Please Enter No.Clusters: "))



# Load and preprocess the dataset
dataset = load_dataset('iris_dataset.csv')
processed_dataset = preprocess_dataset(dataset)

# Split the dataset into training and test sets
random.shuffle(processed_dataset)
split_index = int(0.8 * len(processed_dataset))
train_data = processed_dataset[:split_index]
test_data = processed_dataset[split_index:]

# Plot the dataset
plot_data(train_data)

# Prepare the true labels
true_labels = [data[-1] for data in test_data]

# Train the KNN model
predictions = k_nearest_neighbors(train_data, test_data, k)

# Evaluate the model
accuracy = evaluate_model(predictions, true_labels)
print("Accuracy:", accuracy)

# Example usage of predict_class_label function
input_data = [5.1, 3.5, 1.4, 0.2]  # Example input data
predicted_class = predict_class_label(train_data, input_data, k)
if(predicted_class==0):
  print("Predicted Class:", predicted_class, "Iris-setosa")
elif(predicted_class==1):
    print("Predicted Class:", predicted_class, "Iris-versicolor")
elif(predicted_class==2):
    print("Predicted Class:", predicted_class, "Iris-virginica")

