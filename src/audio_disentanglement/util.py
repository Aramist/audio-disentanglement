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
