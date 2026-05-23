# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Определим интерфейсный класс слоя


class Layer:
    """
    Базовый класс для всех слоев нейронной сети
    """

    def __init__(self):
        self.training = True

    def forward(self, x):
        """
        Прямое распространение
        """
        raise NotImplementedError

    def backward(self, grad_output):
        """
        Обратное распространение
        """
        raise NotImplementedError

    def train(self):
        """
        Переключение в режим обучения
        """
        self.training = True

    def eval(self):
        """
        Переключение в режим инференса
        """
        self.training = False

    def __call__(self, x):
        return self.forward(x)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# реализуем функции активации


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.input = None

    def forward(self, x):
        """
        Прямое распространение для ReLU

        Args:
            x: входной тензор формы (batch_size, ...)

        Returns:
            выходной тензор той же формы
        """

        self.input = x
        output = np.where(x > 0, x, 0)

        return output

    def backward(self, grad_output):
        """
        Обратное распространение для ReLU

        Args:
            grad_output: градиент от следующего слоя

        Returns:
            градиент для предыдущего слоя
        """

        grad_input = grad_output * np.where(self.input > 0, 1, 0)

        return grad_input


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        """
        Прямое распространение для Sigmoid

        Args:
            x: входной тензор

        Returns:
            выходной тензор той же формы, значения в диапазоне (0, 1)
        """

        self.input = x
        sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        self.output = sigmoid(x)

        return self.output

    def backward(self, grad_output):
        """
        Обратное распространение для Sigmoid

        Args:
            grad_output: градиент от следующего слоя

        Returns:
            градиент для предыдущего слоя
        """

        grad_input = grad_output * self.output * (1 - self.output)

        return grad_input


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        """
        Прямое распространение для Tanh

        Args:
            x: входной тензор

        Returns:
            выходной тензор той же формы, значения в диапазоне (-1, 1)
        """

        self.output = np.tanh(x.astype(np.float64))

        return self.output

    def backward(self, grad_output):
        """
        Обратное распространение для Tanh

        Args:
            grad_output: градиент от следующего слоя

        Returns:
            градиент для предыдущего слоя
        """

        grad_input = 1 - self.output**2

        return grad_output * grad_input


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Добавил Linear и Sequential слои


class Linear(Layer):
    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = bias

        self.weight = np.random.randn(input_size, output_size)

        if self.use_bias:
            self.bias = np.random.randn(output_size)
        else:
            self.bias = None

        # Переменные для сохранения входных данных и градиентов
        self.input = None
        self.grad_weight = None
        self.grad_bias = None

    def forward(self, x):
        """
        Прямое распространение для линейного слоя

        Args:
            x: входной тензор формы (batch_size, input_size)

        Returns:
            выходной тензор формы (batch_size, output_size)
        """

        self.input = x

        output = x @ self.weight

        if self.use_bias:
            output = output + self.bias

        return output

    def backward(self, grad_output):
        """
        Обратное распространение для линейного слоя

        Args:
            grad_output: градиент от следующего слоя формы (batch_size, output_size)

        Returns:
            градиент для предыдущего слоя формы (batch_size, input_size)
        """

        grad_input = grad_output @ self.weight.T

        self.grad_weight = self.input.T @ grad_output

        if self.use_bias:
            self.grad_bias = np.sum(grad_output, axis=0)

        return grad_input

    def update_weights(self, learning_rate=0.01):
        """
        Обновление весов с помощью градиентного спуска
        """

        if self.grad_weight is not None:
            self.weight -= learning_rate * self.grad_weight

        if self.use_bias and self.grad_bias is not None:
            self.bias -= learning_rate * self.grad_bias


class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
        self.layer_outputs = []

    def add(self, layer):
        """
        Добавление слоя в последовательность
        """
        self.layers.append(layer)

    def forward(self, x):
        """
        Прямое распространение через все слои

        Args:
            x: входной тензор

        Returns:
            выходной тензор после прохождения всех слоев
        """

        self.layer_outputs = []

        output = x
        for layer in self.layers:
            output = layer.forward(output)
            self.layer_outputs.append(output)

        return output

    def backward(self, grad_output):
        """
        Обратное распространение через все слои в обратном порядке

        Args:
            grad_output: градиент от следующего слоя

        Returns:
            градиент для предыдущего слоя
        """

        grad = grad_output
        for layer in reversed(self.layers):
            # TODO: Примените backward для текущего слоя
            grad = layer.backward(grad)

        return grad

    def train(self):
        """
        Переключение всех слоев в режим обучения
        """
        super().train()
        for layer in self.layers:
            layer.train()

    def eval(self):
        """
        Переключение всех слоев в режим инференса
        """
        super().eval()
        for layer in self.layers:
            layer.eval()

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        return self.layers[idx]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
