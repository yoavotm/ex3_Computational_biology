import numpy as np
import random


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Convert all 0 labels to -1
def normalize(input):
    return np.where(input == 0, -1, input)

# Getting data from the path argument
def load_data_from_path(path):
    samples = []
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
    # Splitting each strings to bits of 0 or 1
    for line in lines:
        d_string, y = line.strip().split()
        samples.append(np.array([int(bit) for bit in d_string]))
        labels.append(int(y))
    data = list(zip(samples, labels))
    random.shuffle(data)
    samples, labels = zip(*data)
    return np.array(samples), np.array(labels)

def load_data():
    train_path = input("Enter the path to the training data: ")
    train_input, train_label = load_data_from_path(train_path)

    test_path = input("Enter the path to the testing data: ")
    test_input, test_label = load_data_from_path(test_path)

    return normalize(train_input), train_label, normalize(test_input), test_label




def save_state(net, output_file, train_acc, test_acc):
    with open(output_file, 'w') as f:
        # Train Accuracy
        f.write('accuracy:\n')
        f.write(f'Train: {train_acc}\n')
        f.write(f'Test: {test_acc}\n')

        f.write(f'Layer sizes:\n{net.layers_dims}\n')

        f.write('Weights table:\n')
        f.write('Src layer , src neuron , dest layer , dest neuron , weight value\n')
        for idx, layer in enumerate(net.weights):
            for j, neuron in enumerate(layer):
                for k, weight in enumerate(neuron):
                    f.write(f'{idx+1} , {j+1} , {idx+2} , {k+1} , {weight}\n')

        # Network
        f.write('\n')
        f.write('Network:\n')
        f.write(f'{net.weights}\n')
    print(f"Data saved in: {output_file}")