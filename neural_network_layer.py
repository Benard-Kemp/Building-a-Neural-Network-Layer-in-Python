
def neuron_output(inputs, weights, bias):
    total = 0.0
    for x, w in zip(inputs, weights):
        total += x * w
    total += bias
    return total

def relu(z):
    return max(0.0, z)

def dense_layer(inputs, weights_list, bias_list, activation):
    outputs = []

    for weights, bias in zip(weights_list, bias_list):
        z = neuron_output(inputs, weights, bias)
        a = activation(z)
        outputs.append(a)

    return outputs

inputs = [1.0, 2.0]

weights_list = [
    [0.5, -1.0],   # Neuron 1
    [1.0, 1.0],    # Neuron 2
    [-0.5, 2.0]    # Neuron 3
]

bias_list = [0.0, 1.0, -1.0]

outputs = dense_layer(inputs, weights_list, bias_list, relu)
print(outputs)
