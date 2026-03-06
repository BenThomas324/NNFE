
from dataclasses import dataclass
import yaml
from .utils import get_dict, get_Path

@dataclass(frozen=True)
class NetworkConfig:

    name: str
    kwargs: dict
    load_model: str = None
    static: bool = False

    @classmethod
    def from_dict(cls, params):
        name = params["name"]
        kwargs = params["kwargs"]
        load_model = params.get("load_model", None)
        static = params.get("static", False)

        return cls(name=name, 
                   kwargs=kwargs, 
                   load_model=load_model, 
                   static=static)

@dataclass(frozen=True)
class OptimizerConfig:

    name: str
    lr_scheduler: bool = True
    optimizer_kwargs: dict = None
    scheduler: dict = None

    @classmethod
    def from_dict(cls, params):
        name = params["name"]
        lr_scheduler = params.get("lr_scheduler", True)
        optimizer_kwargs = params.get("optimizer_kwargs", None)
        scheduler = get_dict(params, "scheduler")
        return cls(name=name, 
                   lr_scheduler=lr_scheduler,
                   optimizer_kwargs=optimizer_kwargs,
                   scheduler=scheduler)

@dataclass(frozen=True)
class MLConfig:

    networks: dict[str, NetworkConfig]
    optimizer: OptimizerConfig
    epochs: int
    batch_size: int
    rng_key: int | str = 0

    @classmethod
    def from_dict(cls, params):
        networks = {net_key: NetworkConfig.from_dict(net_params) for net_key, net_params in params["networks"].items()}
        optimizer = OptimizerConfig.from_dict(params["optimizer"])
        rng_key = params.get("rng_key", 0)
        if rng_key is None:
            rng_key = 0

        return cls(networks=networks, 
                   optimizer=optimizer, 
                   epochs=params["epochs"], 
                   batch_size=params["batch_size"],
                   rng_key=rng_key)

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            params = yaml.safe_load(f)

        return cls.from_dict(params)