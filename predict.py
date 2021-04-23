import argparse
from typing import Optional, List

from gradient_descent import GradientDescent, NormalizationParams
from utils import get_normalization_params_from_file


def read_dataset(path: str, columns: List[str], separator: str = ',', verbose: bool = False) -> List[List[float]]:
    if verbose:
        print(f'В файле по пути "{path}" ожидается разделитель `{separator}`')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
    except Exception as ex:
        raise ValueError(f"Мы не смогли открыть и прочитать файл по пути {path} ({ex})")
    if len(lines) < 2:
        raise ValueError("В файле меньше 2 строк, поэтому либо файл некорректен, либо в нём нет данных")
    header = lines[0].split(separator)
    if len(set(header)) < len(header):
        raise ValueError(f"В header-е есть одинаковые названия столбцов")
    header_length = len(header)
    if header_length != len(columns):
        raise ValueError(f"В header-е должно быть столько же столбцов, "
                         f"сколько было фичей во входном датасете {len(columns)} "
                         f"(возможно, разделитель в файде отличается от {separator})")
    assert set(header) == set(columns), "Названия столбцов с фичами во входном файле и " \
                                        "столбцов в тестовом датасете отличаются"
    feature_arrays = []
    for i, line in enumerate(lines[1:], start=1):
        values = line.split(separator)
        if header_length != len(values):
            raise ValueError(f"Количество значений в header-е и в строке {i} отличается. В header-е {header_length}, "
                             f"а в строке {i} {len(values)}")
        try:
            values = list(map(float, values))
        except ValueError as ex:
            raise ValueError(f"Все значения в строке {i} невозможно привести к типу float ({ex})")
        feature_arrays.append([values[header.index(column)] for column in columns])
    return feature_arrays


def main(path: str, use_input: bool, separator: str = ',', file_with_params: Optional[str] = "params.json",
         verbose: bool = False) -> None:
    gd = GradientDescent(verbose=verbose)
    if file_with_params is None:
        if verbose:
            print("Файл с коэффициентами не был передан, поэтому все коэффициенты приравниваются к нулю")
        print(f'Предсказанные целевые переменные машин для датасета по пути "{path or ""}": 0.0')
        return
    dct: NormalizationParams = get_normalization_params_from_file(file_with_params)
    if use_input:
        while True:
            values: List[float] = []
            for column in dct.columns:
                try:
                    values.append(float(input(f"Введите значение фичи '{column}' машины: ")))
                except ValueError as ex:
                    raise ValueError(f"Было введено некорректное значение пробега машины ({ex})")
            result = gd.predict([values], file_with_params=file_with_params)[0]
            print(f"Предсказанная {dct.target_column} машины: {result}")
            not_to_continue = input("Если хотите продолжить, нажмите `Enter`: ")
            if not_to_continue:
                break
    else:
        feature_arrays: List[List[float]] = read_dataset(path, dct.columns, separator=separator, verbose=verbose)
        result = gd.predict(feature_arrays, file_with_params=file_with_params)
        print(f'Предсказанные {dct.target_column} машин для датасета по пути "{path}": {result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Предсказание цены машины на основе данных о её характеристиках с помощью линейной регрессии.'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--path', type=str, help='путь к датасету с данными')
    group.add_argument('-i', '--input', action='store_true', help='ввод данных с клавиутары')
    parser.add_argument('-s', '--separator', type=str, default=',', help='разделитель в файле с датасетом')
    parser.add_argument('-f', '--file_with_params', type=str,
                        help='путь к файлу, где хранятся значения обученной модели')
    parser.add_argument('-v', '--verbose', action='store_true', help='подробный вывод')
    args = parser.parse_args()
    try:
        main(args.path, args.input, args.separator, args.file_with_params, args.verbose)
    except Exception as e:
        print(f'Произошла ошибка: {e}')
