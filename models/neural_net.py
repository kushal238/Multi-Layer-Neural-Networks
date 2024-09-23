"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initializing the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        self.m = {}
        self.v = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])
            # self.params["Out" + str(i)] = []
            if opt == "Adam":
              self.m["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
              self.m["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])

              self.v["W" + str(i)] = np.zeros_like(self.params["W" + str(i)])
              self.v["b" + str(i)] = np.zeros_like(self.params["b" + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # print(X.shape)
        # print(W.shape)
        # print("Mult:", np.matmul(X, W).shape)
        # print("b:", b.shape)
        b_reshaped =  np.transpose(b.reshape(-1,1))
        X_ones = np.ones(X.shape[0]).reshape(-1,1)
        # print((np.matmul(X, W) + np.matmul(X_ones, b_reshaped)).shape)
        lin_out = np.matmul(X, W) + np.matmul(X_ones, b_reshaped) # if X is N x D and W is D x hidden_size => N x hidden_size
        return lin_out

        
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        temp_ones = np.transpose(np.ones(X.shape[0]).reshape(-1,1))
      
        de_dw = np.matmul(np.transpose(X), de_dz) + reg*W/N
        de_db = np.matmul(temp_ones, de_dz)
        de_dx = np.matmul(de_dz, np.transpose(W))
        return (de_dw,de_db,de_dx)

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # print(X)
        # print(np.maximum(X,0))
        return np.maximum(X,0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        return np.where(X > 0, 1, np.where(X <= 0, 0, X))

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-1*x))

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        return np.multiply(self.sigmoid(X), self.sigmoid(-1*X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.sum((y - p)**2)/(y.shape[0]*y.shape[1])
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return -2 * (y - p)/(y.shape[0]*y.shape[1])
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.multiply(self.mse_grad(y,p), self.sigmoid_grad(self.outputs["OutLin" + str(self.num_layers)]))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computing the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        self.outputs["OutRelu" + str(0)] = X
        layer_in = X
        for i in range(1, self.num_layers + 1):
          lin_out = self.linear(self.params["W" + str(i)], layer_in, self.params["b" + str(i)])
          self.outputs["OutLin" + str(i)] = lin_out
          
          if i is not self.num_layers:
            self.outputs["OutRelu" + str(i)] = self.relu(lin_out)
            layer_in = self.outputs["OutRelu" + str(i)]
          else:
            self.outputs["OutSig"] = self.sigmoid(lin_out)
        return self.outputs["OutSig"]

    def backward(self, y: np.ndarray) -> float:
        """Performing back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # reg = y.shape[0] * y.shape[1]
        back_in = self.mse_sigmoid_grad(y, self.outputs["OutSig"])
        # print("init", back_in.shape)
        # print(back_in)
        de_dw, de_db, de_dx = self.linear_grad(self.params["W" + str(self.num_layers)], self.outputs["OutRelu" + str(self.num_layers-1)], self.params["b" + str(self.num_layers)], back_in, 0.5,self.outputs["OutRelu" + str(self.num_layers-1)].shape[0])
        self.gradients["W" + str(self.num_layers)] = de_dw
        self.gradients["b" + str(self.num_layers)] = de_db
        back_in = de_dx
        # print("Down:", back_in.shape)

        for i in range(self.num_layers - 1, 0, -1):
          relu_local = self.relu_grad(self.outputs["OutLin" + str(i)])
          # print("RElu", relu_local.shape)
          # print(back_in.shape)
          relu_downstream = back_in * relu_local
          de_dw, de_db, de_dx = self.linear_grad(self.params["W" + str(i)], self.outputs["OutRelu" + str(i-1)], self.params["b" + str(i)], relu_downstream, 0.5, self.outputs["OutRelu" + str(i-1)].shape[0])
          self.gradients["W" + str(i)] = de_dw
          self.gradients["b" + str(i)] = de_db          
          back_in = de_dx
        
        return self.mse(y, self.outputs["OutSig"])

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Updating the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # handles updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
          for i in range(1, self.num_layers + 1):
            self.params["W" + str(i)] -= lr*self.gradients["W" + str(i)]
            self.params["b" + str(i)] -= lr*self.gradients["b" + str(i)]
        else:
          for i in range(1, self.num_layers + 1):
            self.m["W" + str(i)] = (b1 * self.m["W" + str(i)]) + ((1 - b1)*self.gradients["W" + str(i)])
            self.m["b" + str(i)] = (b1 * self.m["b" + str(i)]) + ((1 - b1)*self.gradients["b" + str(i)])

            self.v["W" + str(i)] = (b2 * self.v["W" + str(i)]) + ((1 - b2) * self.gradients["W" + str(i)])
            self.v["b" + str(i)] = (b2 * self.v["b" + str(i)]) + ((1 - b2) * self.gradients["b" + str(i)])

            self.params["W" + str(i)] -= lr*self.m["W" + str(i)]/(self.v["W" + str(i)] + eps)
            self.params["b" + str(i)] -= lr*self.m["b" + str(i)]/(self.v["b" + str(i)] + eps)
        return