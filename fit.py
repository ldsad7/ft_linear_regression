import argparse
from typing import List, Tuple

from gradient_descent import GradientDescent


def read_dataset(path: str, verbose: bool = False, separator: str = ',',
                 number_of_target_feature: int = -1) -> Tuple[List[List[float]], List[float], List[str], str]:
    if verbose:
        print(f'В файле по пути "{path}" ожидается разделитель `{separator}`')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception as ex:
        raise ValueError(f"Мы не смогли открыть и прочитать файл по пути {path} ({ex})")
    if len(lines) < 2:
        raise ValueError("В файле меньше 2 строк, поэтому либо файл некорректен, "
                         "либо в нём нет данных")
    header = lines[0].split(separator)
    if len(set(header)) < len(header):
        raise ValueError(f"В header-е есть одинаковые названия столбцов")
    header_length = len(header)
    if header_length < 2:
        raise ValueError(f"В header-е не может быть меньше 2 столбцов, либо разделитель отличается от {separator}, "
                         "либо количество столбцов в header-е меньше минимального количества (2)")
    try:
        target_feature = header[number_of_target_feature]
    except IndexError as ex:
        raise ValueError(f'Передано некорректное значение параметра '
                         f'"number_of_target_feature"={number_of_target_feature} ({ex})')
    number_of_target_feature = (number_of_target_feature + len(header)) % len(header)
    columns: List[str] = [
        column for i, column in enumerate(header)
        if i != number_of_target_feature
    ]
    target_column: str = header[number_of_target_feature]
    if verbose:
        print(f'Целевой переменной является "{target_feature}"')
    target_values = []
    feature_arrays = []
    for i, line in enumerate(lines[1:], start=1):
        values = line.split(separator)
        if header_length != len(values):
            raise ValueError(f"Количество значений в header-е и в строке {i} отличается. "
                             f"В header-е {header_length}, а в строке {i} {len(values)}")
        try:
            values = list(map(float, values))
        except ValueError as ex:
            raise ValueError(f"Все значения в строке {i} невозможно привести к типу float ({ex})")
        target_value = values[number_of_target_feature]
        feature_values = values[:number_of_target_feature] + values[number_of_target_feature + 1:]
        target_values.append(target_value)
        feature_arrays.append(feature_values)
    return feature_arrays, target_values, columns, target_column


def main(path: str, verbose: bool = False, separator: str = ',', number_of_target_feature: int = -1,
         file_with_params: str = "params.json") -> None:
    feature_arrays, target_values, columns, target_column = read_dataset(
        path, verbose=verbose, separator=separator, number_of_target_feature=number_of_target_feature
    )
    gd = GradientDescent(verbose=verbose, columns=columns, target_column=target_column)
    gd.fit(feature_arrays, target_values, file_with_params=file_with_params)
    # print(f'predictions on [[240000], [139800], [61789]]: {gd.predict([[240000], [139800], [61789]])}')
    # print(f"MSE: {gd.count_cost()}")
    # if len(columns) == 1:
    #     from plot_2d_graph import plot_2d_graph
    #     plot_2d_graph(feature_arrays, target_values, file_with_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Обучение линейной регрессии для предсказания цены машины по её пробегу '
                    'на основе предоставленных данных.'
    )
    parser.add_argument('path', type=str, help='путь к датасету с данными')
    parser.add_argument('-v', '--verbose', action='store_true', help='подробный вывод')
    parser.add_argument('-s', '--separator', type=str, default=',', help='разделитель в файле с датасетом')
    parser.add_argument('-f', '--file_with_params', type=str, default='params.json',
                        help='путь к файлу, куда нужно сохранить значения для обученной модели')
    parser.add_argument(
        '-n', '--number_of_target_value', type=int, default=-1,
        help='номер целевой переменной в датасете '
             '(отсчёт идёт с нуля, можно подавать отрицательные значения)'
    )
    args = parser.parse_args()
    try:
        main(
            args.path, verbose=args.verbose, separator=args.separator,
            number_of_target_feature=args.number_of_target_value,
            file_with_params=args.file_with_params
        )
    except Exception as e:
        print(f'Произошла ошибка: {e}')
