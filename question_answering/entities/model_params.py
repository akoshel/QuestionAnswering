from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema
from .data_params import DataParams
from .train_params import TrainParams

@dataclass
class ModelParams:
    data_params: DataParams
    train_params: TrainParams

ModelParamsSchema = class_schema(ModelParams)


def read_model_params(path: str) -> ModelParams:
    with open(path, "r") as input_stream:
        schema = ModelParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
