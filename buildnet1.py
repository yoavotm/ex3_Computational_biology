import copy
import random
import numpy as np
from helpers import *
from Config_nn1 import *


# Neural net class


class Net:
    def __init__(self, layers_sizes, weights=[], init=True):
        self.layers_dims = layers_sizes
        # Giving option to initialize net with predefined weights instead off random weights
        if init:
            self.weights = self.init_net()
        else:
            self.weights = weights

    # Initialize the net's weights with Xavier initialization
    def init_net(self):
        nn = []
        for idx in range(len(self.layers_dims) - 1):
            lmt = np.sqrt(
                6 / (self.layers_dims[idx] + self.layers_dims[idx + 1]))
            weights = np.random.uniform(-lmt, lmt, size=(
                self.layers_dims[idx], self.layers_dims[idx + 1]))
            weights *= 1 - REG
            nn.append(weights)
        return nn

    def forward_pass(self, data):
        if data.ndim == 1:
            data = np.reshape(data, (1, -1))
        for i in range(len(self.weights)):
            updated_inputs = []
            for j in range(self.weights[i].shape[1]):
                # Extract weights for the jth neuron in the current layer
                weights = np.reshape(self.weights[i][:, j], (-1,))
                # Compute dot product
                product = np.dot(data, weights)
                updated_inputs.append(sigmoid(product))
            data = np.array(updated_inputs).T
        return data

    # Calculating fitness score for a neural net
    def fitness_function(self, data, labels, test=False):
        count = 0
        # Perform forward propagation
        outputs = self.forward_pass(data)
        binar_preds = (outputs > 0.5).astype(int)
        preds = binar_preds.flatten()
        # Checking if the predictions were correct
        for label, y in zip(labels, preds):
            if label == y:
                count += 1
        score = count / len(preds)
        # Regularization only if we in train stage
        if test == False:
            reg = 0
            for layer in self.weights:
                reg += np.sum(layer ** 2)
            reg *= (REG / (2 * len(preds)))
            score -= reg
        return score
    
    def classify(self, data):
        outputs = self.forward_pass(data)
        binar_preds = (outputs > 0.5).astype(int)
        preds = binar_preds.flatten()
        return preds

# Writing the best neural net and it's accuracy to output file




def population_initialization():
    global POPULATION_SIZE, POPULATION_ARRAY
    for _ in range(POPULATION_SIZE):
        layer_sizes = [8]
        layer_sizes.insert(0, 16)
        layer_sizes.append(1)
        nn = Net(layer_sizes)
        POPULATION_ARRAY.append((nn, 0))

# Creating one dimensional list from multidimensional list


def flat(list):
    new_list = []
    # Going over every item in the list adding them one after another
    for layer in list:
        for layer2 in layer:
            for i in layer2:
                new_list.append(i)
    return new_list

# Creating multidimensional list from one dimensional list


def re_shape(list, dimensions):
    lst = []
    counter = 0
    for d1, d2 in zip(dimensions[:-1], dimensions[1:]):
        shaped = np.array(list[counter: counter + (d1 * d2)]).reshape((d1, d2))
        counter += d1 * d2
        lst.append(shaped)
    return lst


class Gen:
    def __init__(self):
        pass

    def mutation(self, mutations):
        global DELTA, POPULATION_ARRAY, train_labels, train_inputs
        for net, score in mutations:
            for weight in net.weights:
                # Creating random array of 0 or 1
                mutation = np.random.choice(
                    [0, 1], size=weight.shape, p=[0, 1])
                for i, row in enumerate(weight):
                    for j, k in enumerate(row):
                        # If the item in the random array that matches a weight has 1 as value tahn mutate by
                        # adding random value to the weight
                        if mutation[i, j]:
                            weight[i, j] += np.random.uniform(-DELTA, DELTA)
            weights_fit = net.fitness_function(train_inputs, train_labels)
            POPULATION_ARRAY.append((net, weights_fit))

    # Adding crossovers between random nets to the population
    def crossover(self, crossovers_number):
        global POPULATION_ARRAY, train_inputs, train_labels, test_inputs, test_labels, CROSSOVERS_COUNT
        for i in range(crossovers_number):
            par1, par2 = random.sample(POPULATION_ARRAY, 2)
            p1_weights = flat(par1[0].weights)
            p2_weights = flat(par2[0].weights)
            # Taking the more fit parent as first parent
            if par1[1] > par2[1]:
                first_parent = 1
                child = copy.deepcopy(p1_weights)
                sizes = par1[0].layers_dims
            else:
                first_parent = 2
                child = copy.deepcopy(p2_weights)
                sizes = par2[0].layers_dims
            # Choosing random index. The child will have the weights of first parent until this index, and weights
            # of second parent after this index
            the_chosen_one = np.random.randint(len(p1_weights) - 1)
            if first_parent == 1:
                child[:the_chosen_one] = copy.deepcopy(
                    p2_weights[:the_chosen_one])
            else:
                child[:the_chosen_one] = copy.deepcopy(
                    p1_weights[:the_chosen_one])
            child = re_shape(child, sizes)
            final_offspring = Net(sizes, child, False)
            POPULATION_ARRAY.append(
                (final_offspring, final_offspring.fitness_function(train_inputs, train_labels)))

    def lemarkian_helper(self, mutations, n=3):
        global DELTA, POPULATION_ARRAY, train_labels, train_inputs
        for net, best_fintness_score in mutations:
            for _ in range(n):
                for weight in net.weights:
                    rnd = np.random.choice(
                        [0, 1], size=weight.shape, p=[0.9, 0.1])
                    delta = np.random.uniform(-DELTA, DELTA, size=weight.shape)
                    weight += rnd * delta
                weights_fit = net.fitness_function(train_inputs, train_labels)
                # check if the best score of this generation is the best so far. if so, remember the solution
                if weights_fit > best_fintness_score:
                    best_fintness_score = weights_fit
            POPULATION_ARRAY.append((net, best_fintness_score))

    # This genetic algorithm finds the neural nets with best weights instead of using back-propagation

    def run_genetic_algorithm(self):
        counter_gens_unchanged = 0
        best = float('inf')
        global NUMBER_OF_GENERATIONS, POPULATION_ARRAY, POPULATION_SIZE, MAX_WITHOUT_CHANGE, MUTATIONS_COUNT
        # The loop executes the NUMBER_OF_GENERATIONS of the the genetic algorithm
        for i in range(NUMBER_OF_GENERATIONS):
            POPULATION_ARRAY = sorted(
                POPULATION_ARRAY, key=lambda x: x[1], reverse=True)
            best_found = POPULATION_ARRAY[:10]
            self.lemarkian_helper(best_found)
            self.crossover(CROSSOVERS_COUNT)
            list_mut = random.sample(POPULATION_ARRAY, k=MUTATIONS_COUNT)
            self.mutation(list_mut)
            # Sorting by fitness and taking best population, decreasing order
            POPULATION_ARRAY = sorted(
                POPULATION_ARRAY, key=lambda x: x[1], reverse=True)
            POPULATION_ARRAY = POPULATION_ARRAY[:POPULATION_SIZE]
            print(i + 1, POPULATION_ARRAY[0][1])
            if POPULATION_ARRAY[0][1] == best:
                counter_gens_unchanged += 1
            else:
                counter_gens_unchanged = 0
            best = POPULATION_ARRAY[0][1]
            # If no change for to long we return te best score and without completing the maximal NUMBER_OF_GENERATIONS number
            if counter_gens_unchanged > MAX_WITHOUT_CHANGE:
                return POPULATION_ARRAY[0]
        # Return best net
        return POPULATION_ARRAY[0]


def run_genetic_algorithm():
    best_score = 0
    # Calculating fitness for the random nets for the first time
    for i in range(len(POPULATION_ARRAY)):
        current_net = POPULATION_ARRAY[i][0]
        current_fitness_score = current_net.fitness_function(
            train_inputs, train_labels)
        if current_fitness_score > best_score:
            best_score = current_fitness_score
        POPULATION_ARRAY[i] = (current_net, current_fitness_score)
    print(f"0 {best_score}")
    genetic_algorithm_instance = Gen()
    best_net = genetic_algorithm_instance.run_genetic_algorithm()
    fitness_score = best_net[0].fitness_function(
        test_inputs, test_labels, True)
    save_state(best_net[0], OUTPUT_DIR, best_net[1], fitness_score)
    print(f"Maximal accuracy for train: {best_net[1]}")
    print(f"Accuracy for test: {fitness_score}")


def data_init():
    global train_inputs, train_labels, test_inputs, test_labels
    train_inputs, train_labels, test_inputs, test_labels = load_data()


def main():
    population_initialization()

    data_init()

    run_genetic_algorithm()


if __name__ == '__main__':
    main()
