import dataclasses
import json
from typing import List


@dataclasses.dataclass
class NormalizationParams:
    coefs: List[float]
    feature_mins: List[float]
    feature_maxs: List[float]
    min_target_value: float
    max_target_value: float
    columns: List[str]
    target_column: str


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def get_normalization_params_from_file(params_file: str) -> NormalizationParams:
    try:
        with open(params_file, 'r', encoding='utf-8') as f:
            normalization_params: NormalizationParams = NormalizationParams(**json.load(f))
    except Exception as ex:
        raise ValueError(f'Не смогли прочесть содержимое файла по пути "{params_file or ""}" ({ex})')
    return normalization_params
