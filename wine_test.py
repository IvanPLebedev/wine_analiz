import csv
from typing import List, Callable
from util import normalize_by_feature_scaling, add_one_to_nList
from network import Network
from random import shuffle

def wine_interpret_output(output: List[float]) -> int:
    max_elem = max(output)
    index = output.index(max_elem)
    return index + 3
    
def network_test(network_dimension_list: List[int], learning_rate: float,
    trainers: List[List[float]], trainers_correct: List[List[float]],
    testers: List[List[float]], testers_corrects: List[int],
    interpret_output: Callable[[List[float]], int]) -> float:
    wine_network: Network = Network(network_dimension_list, learning_rate)
    for _ in range(10):
        wine_network.train(trainers, trainers_correct)
    print(f'тест нейросети размерностью {network_dimension_list} и скоростью обучения {learning_rate}')
    wine_results = wine_network.validate(testers, testers_corrects, interpret_output)
    print(f"{wine_results[0]} правильно из {wine_results[1]} = {wine_results[2] * 100}%")
    return wine_results[2]

#возвращает вероятности успеха для конкретных карт нейронов в сети
def test(network_lists: List[List[int]]) -> List[float]:
    wine_parameners: List[List[float]] = []
    wine_quality: List[int] = []
    wine_quality_list: List[List[float]] = []
    with open('winequality-red.csv', mode='r') as winequality_file:
        wines: List = list(csv.reader(winequality_file))
        shuffle(wines)
        for wine in wines:
            parameters: List[float] = [float(n) for n in wine[0:11]]
            wine_parameners.append(parameters)
            quality: int = int(wine[11])
            wine_quality.append(quality)
            quality_list: List[float] = []
            for _ in range(6):
                quality_list.append(0.0)
            quality_list[quality - 3] = 1.0
            wine_quality_list.append(quality_list)
    normalize_by_feature_scaling(wine_parameners)

    wine_trainers: List[List[float]] = wine_parameners[0:1499]
    wine_trainers_correct: List[List[float]] = wine_quality_list[0:1499]
    wine_testers: List[List[float]] = wine_parameners[1499:]
    wine_testers_corrects: List[int] = wine_quality[1499:]
    
    best_network_list: List[int] = []
    networks_res: List[float] = []
    best_res = 0.0
    for network_list in network_lists:
            res = network_test(network_list, 0.3, wine_trainers, wine_trainers_correct,
            wine_testers, wine_testers_corrects, wine_interpret_output)
            networks_res.append(res)
            if res > best_res:
                best_network_list = network_list
                best_res = res
    print(f'лучший результат теста: {best_network_list} с вероятностью успеха {best_res}')  
    return networks_res


if __name__ == "__main__":
    network_lists: List[List[int]] = []
    test_list: List[int] = [5]
    for _ in range(10):
        test_list = add_one_to_nList(test_list, 15, 5)
        test_list2 = list(test_list)
        test_list2.append(6)
        for n in range(8, 12):
            network_list = list(test_list2)
            network_list.insert(0, n)
            network_lists.append(network_list)
        
    networks_res: List[float] = []
    x = 10 # количество тестов
    for i in range(x):
        print(f'тест №{i}')
        res = test(network_lists)
        if i == 0:
            networks_res = res
        else:
            for j in range(len(network_lists)):
                networks_res[j] += res[j]

    m = networks_res.index(max(networks_res))

    print(f'лучший результат всего теста:{network_lists[m]} со средней вероятностью {networks_res[m]/x}')
    # в результате получил лучший результат всего теста:[11, 6, 6] 
    # со средней вероятностью 0.6260000000000001
    