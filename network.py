from __future__ import annotations
from typing import List, Callable, TypeVar, Tuple
from functools import reduce
from layer import Layer
from util import sigmoid, derivative_sigmoid

T = TypeVar('T') # тип выходных данных в интерпретации нейронной сети

class Network:
    def __init__(self, layer_structure: List[int], learning_rate: float, activation_function: Callable[[float], float] = sigmoid, derivative_activation_function: Callable[[float], float] = derivative_sigmoid) -> None:
        if len(layer_structure) < 3:
            raise ValueError("Error: Should be at least 3 layers (1 input, 1 hidden, 1 output)")
        self.layers: List[Layer] = []
        # входной слой
        input_layer: Layer = Layer(None, layer_structure[0], learning_rate, activation_function, derivative_activation_function)
        self.layers.append(input_layer)
        # скрытые слои и выходной слой
        for previous, num_neurons in enumerate(layer_structure[1::]):
            next_layer = Layer(self.layers[previous], num_neurons, learning_rate, activation_function, derivative_activation_function)
            self.layers.append(next_layer)

    # перемещает входные данные на первый слой, и затем выводит их с первого
    # слоя и подает на второй слой в качестве входных данных, со второго слоя
    # на третий и тд
    def outputs(self, input: List[float]) -> List[float]:
        return reduce(lambda inputs, layer: layer.outputs(inputs), self.layers, input)
    
    # Определяет изменения каждого нейрона на основании ошибок выходных данных
    # по сравнению с ожидаемыми данными
    def backpropagate(self, expected: List[float]) -> None:
        # вычисление дельты для нейронов выходного слоя
        last_layer: int = len(self.layers) - 1
        self.layers[last_layer].calculate_deltas_for_output_layer(expected)
        # вычисление дельты для скрытых слоев в обратном порядке
        for l in range(last_layer - 1, 0, -1):
            self.layers[l].calculate_deltas_for_hidden_layer(self.layers[l + 1])
    
    # сама функция backpropagate() не изменяет веса
    # функция upgrade_weights() использует дельты, вычисленные в backpropagate(),
    # чтобы действительно изменить веса
    def update_weights(self) -> None:
        for layer in self.layers[1:]: # пропустить входной слой
            for neuron in layer.neurons:
                for w in range(len(neuron.weights)):
                    neuron.weights[w] = neuron.weights[w] + (neuron.learning_rate * (layer.previous_layer.output_cache[w]) * neuron.delta)
    
    # train() использует результаты выполнения функции outputs() для нескольких
    # входный данных, сравнивая их с ожидаемыми результатами и передает
    # полученное backpropagate() и update_weights()
    def train(self, inputs: List[List[float]], expecteds: List[List[float]]) -> None:
        for location, xs in enumerate(inputs):
            ys: List[float] = expecteds[location]
            outs: List[float] = self.outputs(xs)
            self.backpropagate(ys)
            self.update_weights()

    # для параметризованнх результатов, которые требуют классификации, эта
    # функция возвращает правильное количество попыток и процентное отношение
    # по сравнению с общим количеством
    def validate(self, inputs: List[List[float]], expecteds: List[T], interpret_output: Callable[[List[float]], T]) -> Tuple[int, int, float]:
        correct: int = 0
        for input, expected in zip(inputs, expecteds):
            result: T = interpret_output(self.outputs(input))
            if result == expected:
                correct += 1
        percentage: float = correct / len(inputs)
        return correct, len(inputs), percentage