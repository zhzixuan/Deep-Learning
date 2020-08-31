import numpy as np
from datasets.toy_data import xor_3_input
# from datasets import toy_data as xor_3_input

def sigmoid(x):
    """
    Compute the sigmoid function.
    """
    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE
    return s


def sigmoid_grad(x):
    """
    Compute the gradient for the sigmoid function.
    """
    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    ### END YOUR CODE
    return ds


def init_layers(input_dim1=3, output_dim1=3,
               input_dim2=3, output_dim2=1,
               seed=1234):
    """
    Initialize parameters for a two-layer neural network.
    Note: Use numpy.ramdon.randn to assign values to weight and bias.
    Args:
        - input_dim1: Input dimension for the first layer
        - output_dim1: Output dimension for the first layer
        - input_dim2: Input dimension for the second layer
        - output_dim2: Output dimension for the second layer
        - seed: Make result reproducible.
    Return:
        Dictionary containing the weight and bias parameters.
            {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        - W: Numpy array of shape (output_dim, input_dim)
        - b: Numpy array of shape (output_dim, 1)
    """
    np.random.seed(seed)

    assert output_dim1 == input_dim2, "output dimension 1 should equal to input dimension 2"

    ### YOUR CODE HERE (4 lines)
    W1 = np.random.randn(output_dim1, input_dim1) 
    # b1 = np.zeros((output_dim1, 1))
    b1 = np.random.randn(output_dim1, 1)
    W2 = np.random.randn(output_dim2, input_dim2) 
    # b2 = np.zeros((output_dim2, 1))
    b2 = np.random.randn(output_dim2, 1)
    ### END YOUR CODE

    layer_params = {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2
    }
    return layer_params


def forward_propagation(X, layer_params):
    """
    Forward propagation for neural network.
    Args:
        - X: Input for neural network, with shape (n_features, batch_size).
        - layer_params: Result of function init_layers().
    Return:
        - A2: Output of neural network, with shape (1, batch_size).
        - cache: Results from different layers, which are further used for back propagation.
    """
    # Get W1, b1, W2, b2 from layer_params
    ### YOUR CODE HERE (4 lines)
    W1 = layer_params["W1"]
    b1 = layer_params["b1"]
    W2 = layer_params["W2"]
    b2 = layer_params["b2"]
    ### END YOUR CODE

    # Compute Z1, A1, Z2, A2
    # Hint: to compute W*X + b for numpy.ndarray, you can use W.dot(X) + b
    ### X.shape  : (n_features, batch_size) = (input_dim1, batch_size)
    ### W1.shape : (output_dim1, input_dim1)
    ### b1.shape : (output_dim1, 1)
    ### Z1.shape : (output_dim1, batch_size)
    ### A1.shape : (output_dim1, batch_size)
    ### W2.shape : (output_dim2, input_dim2) = (output_dim2, output_dim1)
    ### b2.shape : (output_dim2, 1)
    ### Z2.shape : (output_dim2, batch_size)
    ### A2.shape : (output_dim2, batch_size)

    ### YOUR CODE HERE (~4 lines)
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    ### END YOUR CODE

    cache = {
        "X": X,
        "A1": A1,
        "Z1": Z1,
        "Z2": Z2
    }

    return A2, cache


def backward_propagation(Y, Y_hat, cache, layer_params):
    """
    Backward propagation for neural network.
    Args:
        - Y: Input for neural network, with shape (n_features, batch_size).
        - Y_hat: with shape (1, batch_size). Note, Y_hat equals to A2.
        - cache: Cache from forward propagation.
        - layer_params: Result of function init_layers().
    Return:
        - Gradients for trainable parameters:
            {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    """
    # Get batch size m from Y

    ### YOUR CODE HERE (1 lines)
    m = Y.shape[1]
    ### END YOUR CODE

    # Compute dZ2, dW2, db2, dZ1, dW1, db1 from your calculation.
    # Hint: Some values are ready from layer_params and cache.

    ### YOUR CODE HERE
    W2 = layer_params["W2"]
    A1 = cache["A1"]
    Z1 = cache["Z1"]
    X = cache["X"]

    dZ2 = Y_hat - Y    
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_grad(Z1)
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    ### END YOUR CODE

    grads = {
        "W1": dW1, "b1": db1,
        "W2": dW2, "b2": db2,
        "Z1": dZ1, "Z2": dZ2
    }

    return grads


def parameter_update(grads, layer_params, learning_rate):
    """
    Update parameters, i.e., W = W - learning_rate * dW
    Args:
        - grads: Gradients computed from backword function.
        - layer_params: Result of function init_layers.
        - learning_rate: Floating point scalar.
    Return:
        - layer_params: Updated parameters.
    """
    ### YOUR CODE HERE
    layer_params["W1"]  = layer_params["W1"] - learning_rate * grads["W1"]
    layer_params["b1"]  = layer_params["b1"] - learning_rate * grads["b1"]
    layer_params["W2"]  = layer_params["W2"] - learning_rate * grads["W2"]
    layer_params["b2"]  = layer_params["b2"] - learning_rate * grads["b2"]
    ### END YOUR CODE
    return layer_params


def cost_fn(Y_hat, Y):
    """
    Compute loss given predicted values and true values:
        loss = -1. * sum(y * log(y_hat) + (1-y) * log(1-y_hat))
    Args:
        - Y_hat: Output of neural network, numpy array of shape (1, batch_size)
        - Y: True values, numpy array of shape (1, batch_size)
    Return:
        - loss: Cross-entropy loss.
    """
    ### YOUR CODE HERE
    m = Y.shape[1]
    cost = -1./m * np.sum(Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat))
    cost = np.squeeze(cost)
    return cost
    ### END YOUR CODE
    # return NotImplementedError()
    

def train(n_iters=100000, learning_rate=0.2, plot_loss=True):
    """
    Train the neural network using backpropagation algorithm and toy data.
    Args:
        - n_iters: Number of training iterations.
        - learning_rate: Learning rate. Do modify learning rate to see how the loss behaves.
        - plot_loss: Boolean value, True to plot training losses.
    """
    # Load toy data
    X, Y = xor_3_input()
    # Initialize the network layers.
    layer_params = init_layers(seed=1234)
    # Total losses
    losses = []

    for i in range(n_iters):
        # Compute forward propagation
        Y_hat, cache = forward_propagation(X, layer_params)
        # Compute backpropagation
        grads = backward_propagation(Y, Y_hat, cache, layer_params)
        # Update parameters.
        layer_params = parameter_update(grads, layer_params, learning_rate=learning_rate)
        # Compute loss
        loss = cost_fn(Y_hat, Y)
        losses.append(loss)
        # Print out loss every 2500 frequency.
        if i % 2500 == 0:
            print("Iteration {:5d} loss: {:0.6f}".format(i, loss))

    # Plot losses for the entire training process.
    if plot_loss:
        try:
            import matplotlib.pyplot as plt
            plt.plot(losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()
        except EnvironmentError:
            print("matplotlib not found.")


if __name__ == '__main__':
    train(50000)
    # train(1, plot_loss=False)
