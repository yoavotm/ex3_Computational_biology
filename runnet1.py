import numpy as np

# Convert all 0 labels to -1
def normalize(input):
    input[input == 0] = -1
    return input

# Neural network class
class Net:
    def __init__(self, all_layers_sizes, weights):
        self.layer_sizes = all_layers_sizes
        self.weights = weights

    # Compute the dot product of weights and inputs
    def dot_product(self, inputs, weights):
        return np.dot(inputs, weights)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, data):
        if data.ndim == 1:
            data = np.reshape(data, (1, -1))
        for i in range(len(self.weights)):
            updated_inputs = []
            for j in range(self.weights[i].shape[1]):
                # Extract weights for the jth neuron in the current layer
                weights = np.reshape(self.weights[i][:, j], (-1,))
                # Compute dot product
                product = self.dot_product(data, weights)
                updated_inputs.append(self.sigmoid(product))
            data = np.array(updated_inputs).T
        return data

    def classify(self, inputs):
        outputs = self.forward(inputs)
        # Converts the output of the final layer to binary predictions
        binary_predictions = (outputs > 0.5).astype(int)
        return binary_predictions

def compute_accuracy(predictions, labels):
    # Compute the accuracy of the predictions by comparing to the labels
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            correct += 1
    return correct / len(predictions)

def runnet1(weight_file, test_file):
    layer_sizes, network = get_network(weight_file)

    best_solution = Net(layer_sizes, network)

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
    output = [str(out[0]) for out in output]

    with open('sol1.txt', 'w') as f:
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
            break;

    # convert every "array" instance to numpy.ndarray
    network = network.replace("array", "np.array")
    network = eval(network)
    return layer_sizes, network

if __name__ == '__main__':
    print("Enter the testnet file path")
    print("For example: 'testnet1.txt', without the quotes:")
    testnet_file = input()
    runnet1('wnet1.txt', testnet_file)