import numpy as np

# 신경망 클래스
class NeuralNetwork:
    # 초기화
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.3, activation_function='relu'):
        # 활성화 함수 설정
        act_name = activation_function.lower()
        act_func = {
            'sigmoid': (self.sigmoid, self.sigmoid_derivative),
            'relu': (self.relu, self.relu_derivative),
            'leaky_relu': (self.leaky_relu, self.leaky_relu_derivative)
        }
        try:
            self.hidden_act, self.hidden_act_derivative = act_func[act_name]
            print(f'Using activation function: {activation_function}')
        except KeyError:
            raise ValueError(f"Invalid activation function: {activation_function}")
        # 가중치와 편향 초기화
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    # 활성화 함수
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)

    def leaky_relu(self, x):
        return np.maximum(0.01 * x, x)
    
    # 활성화 함수의 미분
    def sigmoid_derivative(self, x):
        y = self.sigmoid(x)
        return y * (1 - y)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, 0.01)

    # 순전파
    def forward(self, X):
        self.X = X
        self.Z1 = self.X @ self.W1 + self.b1
        self.A1 = self.hidden_act(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        # print('X :', X.shape)
        # print('W1 :', self.W1.shape)
        # print('b1 :', self.b1.shape)
        # print('Z1 :', self.Z1.shape)
        # print('A1 :', self.A1.shape)
        # print('W2 :', self.W2.shape)
        # print('b2 :', self.b2.shape)
        # print('Z2 :', self.Z2.shape)
        # print('A2 :', self.A2.shape)
        return self.A2

    # 역전파   
    def backward(self, Y):
        dL_dA2 = self.A2 - Y
        dA2_dZ2 = self.sigmoid_derivative(self.Z2)
        delta2 = dL_dA2 * dA2_dZ2
        dL_dW2 = self.A1.T @ delta2
        dL_db2 = np.sum(delta2, axis=0, keepdims=True)
        dL_dA1 = delta2 @ self.W2.T
        dA1_dZ1 = self.hidden_act_derivative(self.Z1)
        delta1 = dL_dA1 * dA1_dZ1
        dL_dW1 = self.X.T @ delta1
        dL_db1 = np.sum(delta1, axis=0, keepdims=True)
        self.W2 -= dL_dW2 * self.lr
        self.b2 -= dL_db2 * self.lr
        self.W1 -= dL_dW1 * self.lr
        self.b1 -= dL_db1 * self.lr
