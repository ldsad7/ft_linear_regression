import argparse
# unfortunately, method `plot_surface` doesn't work with lists and I don't want to monkey-code it
# so I will use full library
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from fit import read_dataset
from gradient_descent import GradientDescent, NormalizationParams
from utils import get_normalization_params_from_file

NUM_OF_STEPS: int = 100


def plot_3d_graph(feature_arrays: List[List[float]], target_values: List[float],
                  params_file: str, verbose: bool = False) -> None:
    for feature_array in feature_arrays:
        assert len(feature_array) == 2, "Ожидается 3D линейная регрессия, было подано не 2 значения фичей"
    assert len(feature_arrays) == len(target_values), "Число поданных массивов со значениями фичей " \
                                                      "и целевыми значениями отличаются"
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": "3d"})
    xs = [feature_array[0] for feature_array in feature_arrays]
    ys = [feature_array[1] for feature_array in feature_arrays]
    ax.scatter(xs, ys, target_values, color="red", label="Данные")

    dct: NormalizationParams = get_normalization_params_from_file(params_file)
    assert len(dct.feature_mins) == 3 and len(dct.feature_maxs) == 3 and \
           len(dct.columns) == 2, f"Некорректные данные по пути {params_file}"
    xs = np.arange(*ax.get_xlim(), (dct.feature_maxs[1] - dct.feature_mins[1]) / NUM_OF_STEPS)
    ys = np.arange(*ax.get_ylim(), (dct.feature_maxs[2] - dct.feature_mins[2]) / NUM_OF_STEPS)

    gd: GradientDescent = GradientDescent(verbose=verbose)
    zs = gd.predict([[x, y] for x in xs for y in ys], params_file)

    xs, ys = np.meshgrid(xs, ys)
    zs = np.array(zs).reshape(xs.shape)
    surf = ax.plot_surface(xs, ys, zs, label="Плоскость линейной регрессии")
    # https://stackoverflow.com/a/54994985
    try:
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d
    except Exception:
        pass

    ax.set_xlabel(dct.columns[0])
    ax.set_ylabel(dct.columns[1])
    ax.set_zlabel(dct.target_column)
    ax.set_title("Linear Regression (MSE)")
    ax.legend()
    plt.show()


def plot_2d_graph(feature_arrays: List[List[float]], target_values: List[float],
                  params_file: str, verbose: bool = False) -> None:
    for feature_array in feature_arrays:
        assert len(feature_array) == 1, "Ожидается 2D линейная регрессия, было подано больше значений фичей"
    assert len(feature_arrays) == len(target_values), "Число поданных массивов со значениями фичей " \
                                                      "и целевыми значениями отличаются"
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(feature_arrays, target_values, label="Данные", color="red")

    dct: NormalizationParams = get_normalization_params_from_file(params_file)
    assert len(dct.feature_mins) == 2 and len(dct.feature_maxs) == 2 and \
           len(dct.columns) == 1, f"Некорректные данные по пути {params_file}"

    borders = list(map(int, ax.get_xlim()))
    if not NUM_OF_STEPS:
        raise ValueError("Переменная NUM_OF_STEPS не может быть нулевой")
    xs: List[List[float]] = [
        [i] for i in range(*borders, int((dct.feature_maxs[1] - dct.feature_mins[1]) / NUM_OF_STEPS))
    ]

    gd: GradientDescent = GradientDescent(verbose=verbose)
    ys = gd.predict(xs, params_file)

    ax.plot(xs, ys, label="Линия линейной регрессии")

    ax.set_xlabel(dct.columns[0])
    ax.set_ylabel(dct.target_column)
    ax.set_title("Linear Regression (MSE)")
    ax.legend()
    plt.show()


def main(path: str, dimension: int, verbose: bool = False, separator: str = ',', number_of_target_feature: int = -1,
         file_with_params: str = "params.json") -> None:
    feature_arrays, target_values, columns, target_column = read_dataset(
        path, verbose=verbose, separator=separator, number_of_target_feature=number_of_target_feature
    )

    assert len(columns) == dimension - 1, \
        f"Ожидалось, что в файле будет {dimension - 1} фичи, поскольку график {dimension}D"
    if dimension == 2:
        plot_2d_graph(feature_arrays, target_values, file_with_params, verbose=verbose)
    elif dimension == 3:
        plot_3d_graph(feature_arrays, target_values, file_with_params, verbose=verbose)
    else:
        raise ValueError(f"Некорректное значение измерения ({dimension})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='График линейной регрессии для предсказания цены машины по её пробегу '
                    'на основе предоставленных данных.'
    )
    parser.add_argument('path', type=str, help='путь к датасету с данными')
    parser.add_argument('-d', '--dimension', required=True, type=int, help='измерение графика', choices=(2, 3),
                        default=2)
    parser.add_argument('-v', '--verbose', action='store_true', help='подробный вывод')
    parser.add_argument('-s', '--separator', type=str, default=',', help='разделитель в файле с датасетом')
    parser.add_argument('-f', '--file_with_params', type=str, default='params.json',
                        help='путь к файлу, где хранятся значения обученной модели')
    parser.add_argument(
        '-n', '--number_of_target_value', type=int, default=-1,
        help='номер целевой переменной в датасете (отсчёт идёт с нуля, можно подавать отрицательные значения)'
    )
    args = parser.parse_args()
    try:
        main(
            args.path, args.dimension, verbose=args.verbose, separator=args.separator,
            number_of_target_feature=args.number_of_target_value, file_with_params=args.file_with_params
        )
    except Exception as e:
        print(f'Произошла ошибка: {e}')
