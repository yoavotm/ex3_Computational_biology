import numpy as np
from buildnet1 import NN
from Config_nn1 import *
from helpers import *

# Neural network class

def compute_accuracy(predictions, labels):
    # Compute the accuracy of the predictions by comparing to the labels
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1
    return correct / len(predictions)

def runnet1(weight_file, test_file):
    layer_sizes, network = get_network(weight_file)

    best_solution = NN(layer_sizes, network, False)

    with open(test_file, 'r') as file:
        data = file.read().splitlines()
        # get the first part
        input_data = [x.split(' ')[0] for x in data]

    input_data = np.array([list(map(int, x)) for x in input_data])
    # normalize the input data
    for i in range(len(input_data)):
        input_data[i] = normalize(input_data[i])
    output = best_solution.classify(input_data)

    #convert output to '0' and '1' array
    output = ['1' if x == 1 else '0' for x in output]

    with open('sol0.txt', 'w') as f:
        for i in range(len(output)):
            f.write(output[i] + '\n')

def get_network(file_path):
    # Get the Layer sizes array and the Network from the above described file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        if line.startswith("Layer sizes"):
            layer_sizes = list(map(int, lines[lines.index(line) + 1].strip()[1:-1].split(',')))
        elif line.startswith("Network"):
            # get the next line until the end of the file as a string
            network = lines[lines.index(line) + 1:]
            # concat it as a string
            network = ''.join(network)
            break

    # convert every "array" instance to numpy.ndarray
    network = network.replace("array", "np.array")
    network = eval(network)
    return layer_sizes, network

if __name__ == '__main__':
    print("Enter testnet file path, For example: testnet1.txt")
    testnet_file = input()
    runnet1('wnet1.txt', testnet_file)