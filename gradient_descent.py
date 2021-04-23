import json
from typing import List, Optional

from utils import NormalizationParams, EnhancedJSONEncoder


class GradientDescent:
    def __init__(self, verbose: bool = False, columns: List[str] = (), target_column: str = 'unknown'):
        self.verbose = verbose
        self.columns = columns
        self.target_column = target_column

    def _count_cost_values(self) -> List[float]:
        values = []
        for feature_array, target_value in zip(self._feature_arrays, self._target_values):
            values.append(sum(coef * value for coef, value in zip(self.coefs, feature_array)) - target_value)
        return values

    def count_cost(self) -> float:
        """
        counts MSE divided by 2
        """

        values = self._count_cost_values()
        if not self._feature_arrays:
            raise ValueError("Массивы с фичами пустые, невозможноно посчитать MSE")
        return sum(value * value for value in values) / (2 * len(self._feature_arrays))

    def _normalize_feature_arrays(self, feature_mins: List[float], feature_maxs: List[float]) -> None:
        for feature_array in self._feature_arrays:
            for i in range(len(feature_array)):
                if feature_maxs[i] == feature_mins[i]:
                    feature_array[i] = 1.0
                else:
                    feature_array[i] = (feature_array[i] - feature_mins[i]) / (feature_maxs[i] - feature_mins[i])

    def _normalize_target_values(self, min_target_value: float, max_target_value: float) -> None:
        for i in range(len(self._target_values)):
            if max_target_value == min_target_value:
                self._target_values[i] = 1.0
            else:
                self._target_values[i] = (self._target_values[i] - min_target_value) / \
                                         (max_target_value - min_target_value)

    def _normalize_data(self) -> NormalizationParams:
        feature_mins: List[float] = []
        feature_maxs: List[float] = []
        for i in range(len(self._feature_arrays[0])):
            feature_maxs.append(max(feature_array[i] for feature_array in self._feature_arrays))
            feature_mins.append(min(feature_array[i] for feature_array in self._feature_arrays))

        self._normalize_feature_arrays(feature_mins, feature_maxs)

        min_target_value: float = min(self._target_values)
        max_target_value: float = max(self._target_values)

        self._normalize_target_values(min_target_value, max_target_value)

        normalization_params = NormalizationParams(
            self.coefs, feature_mins, feature_maxs, min_target_value, max_target_value, self.columns,
            self.target_column
        )
        return normalization_params

    def fit(self, feature_arrays: List[List[float]], target_values: List[float], coefs: Optional[List[float]] = None,
            learning_rate: float = 0.01, epsilon: float = 10e-7, file_with_params: str = "params.json") -> None:
        assert len(feature_arrays) > 0, "Поданный массив со значениями фич пуст"
        feature_arrays: List[List[float]] = [[1.0] + feature_array for feature_array in feature_arrays]
        target_values = target_values[:]
        assert len(feature_arrays) == len(target_values), \
            "Переданные массивы с фичами и целевыми значениями отличаются по длине"

        if coefs is None:
            if self.verbose:
                print(f"В качестве стартовых коэффициентов заводим массив из нулевых значений")
            coefs = [0.0 for _ in range(len(feature_arrays[0]))]
        assert min([isinstance(coef, (int, float)) for coef in coefs]) is True, \
            "В поданном массиве со значениями фичей есть значения с типом, отличным от float"
        self.coefs = coefs

        for i in range(len(feature_arrays)):
            assert len(self.coefs) == len(feature_arrays[i]), \
                "Количество коэффициентов и длина по крайней мере одного из массивов с фичами отличаются"
        assert min(
            isinstance(value, (int, float)) for feature_array in feature_arrays for value in feature_array) is True, \
            "В поданном массиве со значениями целевых переменных есть значения с типом, отличным от float"
        assert min(isinstance(target_value, (int, float)) for target_value in target_values) is True, \
            "В поданном массиве со значениями целевых переменных есть значения с типом, отличным от float"
        self._feature_arrays = feature_arrays
        self._target_values = target_values

        self._lr: float = learning_rate
        self._epsilon: float = epsilon

        self._normalization_params: NormalizationParams = self._normalize_data()

        while True:
            if self.verbose:
                print(f"current MSE: {self.count_cost()}, making one more step...")
            if self._make_step():
                break

        if self.verbose:
            print(f"finally model is fitted")
            print(f"saving params to the file {file_with_params}")

        self._normalization_params.coefs = self.coefs
        with open(file_with_params, 'w', encoding='utf-8') as f:
            json.dump(self._normalization_params, f, cls=EnhancedJSONEncoder)

    @staticmethod
    def _denormalize_data(values: List[float], min_target_value: float, max_target_value: float) -> List[float]:
        for i in range(len(values)):
            values[i] = values[i] * (max_target_value - min_target_value) + min_target_value
        return values

    def _make_step(self) -> bool:
        if not self._feature_arrays:
            raise ValueError("Массивы с фичами пустые, невозможноно посчитать MSE")
        m = len(self._feature_arrays)
        values = self._count_cost_values()
        need_to_stop = False
        new_coefs: List[float] = []
        for i, coef in enumerate(self.coefs):
            step_size = (self._lr / m) * sum(
                value * feature_array[i] for feature_array, value in zip(self._feature_arrays, values)
            )
            if abs(step_size) < self._epsilon:
                need_to_stop = True
            new_coefs.append(self.coefs[i] - step_size)
        if not need_to_stop:
            self.coefs = new_coefs  # simultaneous update
        return need_to_stop

    def predict(self, feature_arrays: List[List[float]],
                file_with_params: Optional[str] = "params.json") -> List[float]:
        assert len(feature_arrays) > 0, "Поданный массив со значениями фич пуст"
        import numpy as np
        assert min(
            isinstance(value, (int, float, np.floating, np.integer))
            for feature_array in feature_arrays for value in feature_array
        ) is True, "В поданном массиве со значениями целевых переменных есть значения с типом, отличным от float"
        if file_with_params is not None:
            try:
                with open(file_with_params, 'r', encoding='utf-8') as f:
                    normalization_params: NormalizationParams = NormalizationParams(**json.load(f))
            except Exception as ex:
                raise ValueError(f"Мы не смогли распарсить содержимое переданного в метод predict файла ({ex})")
        else:
            try:
                normalization_params = getattr(self, "_normalization_params")
            except AttributeError as e:
                raise ValueError(f"Модель ещё не обучалась ({e})")
        self._feature_arrays: List[List[float]] = [[1.0] + feature_array for feature_array in feature_arrays]
        for i in range(len(self._feature_arrays)):
            assert len(normalization_params.coefs) == len(self._feature_arrays[i]), \
                "Количество коэффициентов и длина по крайней мере одного из массивов с фичами отличаются"
        self._normalize_feature_arrays(normalization_params.feature_mins, normalization_params.feature_maxs)
        target_values: List[float] = [
            sum(coef * value for coef, value in zip(normalization_params.coefs, self._feature_arrays[i]))
            for i in range(len(self._feature_arrays))
        ]
        return self._denormalize_data(
            target_values, normalization_params.min_target_value, normalization_params.max_target_value
        )
