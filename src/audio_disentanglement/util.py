from dataclasses import dataclass, field


@dataclass
class ConfigNamespace:
    num_optimization_steps: int = 1_000_000
    clip_gradients: bool = True
    batch_size: int = 128
    dims_per_aug: dict[str, int] = field(
        default_factory=lambda: {
            "gain": 2,
            "pitch_shifting": 5,
            "time_stretching": 5,
        }
    )
    disentangler_type: str = "triangular"
    disentangler_num_layers: int = 2
    disentangler_nonlinearity: str = "identity"

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "ConfigNamespace":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {
            key: value for key, value in config_dict.items() if key in valid_keys
        }
        # reconstruct the nested dictionary for dims_per_aug if it exists in the config_dict
        return cls(**filtered_dict)
