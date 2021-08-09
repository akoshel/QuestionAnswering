from dataclasses import dataclass

@dataclass
class TrainParams:
    num_epoch: int
    batch_size: int
    learning_rate: float
    model_checkpoint_path: str
