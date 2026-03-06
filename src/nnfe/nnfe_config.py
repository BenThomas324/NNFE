
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml
from cardiax.input_file.config import ProblemConfig
from .ml_config import MLConfig
from .utils import validate_sampler, get_dict, get_Path

# Not sure exactly what separation to do here or if it's necessary
@dataclass(frozen=True)
class ProjectConfig:

    name: str = "Project"
    parent_dir: Path = Path(".")
    save: bool = True
    print_progress: int = None
    save_progress: int = None
    extra_dirs: dict = None
    trained_weights_path: str = None

    @classmethod
    def from_dict(cls, params):
        name = params["name"]
        parent_dir = params["parent_dir"]
        save = params["save"]
        print_progress = int(params["print_progress"])
        save_progress = int(params["save_progress"])
        extra_dirs = get_dict(params, "extra_dirs")
        trained_weights_path = get_Path(params, "trained_weights_path")

        return cls(name=name,
                   parent_dir=parent_dir,
                   save=save,
                   print_progress=print_progress,
                   save_progress=save_progress,
                   extra_dirs=extra_dirs,
                   trained_weights_path=trained_weights_path)

# Add a few plotting things to unpack
# Could possibly move this to CARDIAX-VIS after a while
@dataclass(frozen=True)
class PlotterConfig:

    plot_loss: bool = True
    plot_lr: bool = True
    plot_sample: bool = True

    @classmethod
    def from_dict(cls, params):
        return cls(plot_loss=params["plot_loss"],
                   plot_lr=params["plot_lr"])

# This should hold data regarding the sampling technique
# Other class should do the actual sampling from distribution
@dataclass(frozen=True)
class SamplerConfig:

    training_sampler: str = "uniform"
    training_kwargs: dict = None
    testing_sampler: str = "uniform"
    testing_kwargs: dict = None

    @classmethod
    def from_dict(cls, params):
        training_sampler = validate_sampler(params["training_sampler"])
        testing_sampler = validate_sampler(params["testing_sampler"])


        return cls(training_sampler=training_sampler,
                   testing_sampler=testing_sampler,
                   training_kwargs=params["training_kwargs"],
                   testing_kwargs=params["testing_kwargs"])

@dataclass(frozen=True)
class NNFEParamsConfig:

    natural: dict = None
    essential: dict = None
    natural_order: list = None
    essential_order: list = None

    @classmethod
    def from_dict(cls, params):
        natural = get_dict(params, "natural")
        essential = get_dict(params, "essential")
        natural_order = params["natural_order"]
        essential_order = params["essential_order"]

        return cls(natural=natural,
                   essential=essential,
                   natural_order=natural_order,
                   essential_order=essential_order)

@dataclass(frozen=True)
class NNFEConfig:

    project: ProjectConfig
    plotter: PlotterConfig
    sampler: SamplerConfig
    FE: ProblemConfig
    ML: MLConfig
    NNFE: NNFEParamsConfig

    @classmethod
    def from_yaml(cls, path: Path):
        path = Path(path)
        parent = path.parent
        with open(path) as f:
            params = yaml.safe_load(f)

        if type(params["FE"]) == str and ".yaml" in params["FE"]:
            fe_file = params["FE"]
            with open(parent / fe_file) as f:
                problem_params = yaml.safe_load(f)
        else:
            problem_params = get_dict(params, "FE")

        if type(params["ML"]) == str and ".yaml" in params["ML"]:
            ml_file = params["ML"]
            with open(parent / ml_file) as f:
                ml_params = yaml.safe_load(f)
        else:
            ml_params = get_dict(params, "ML")

        # Unpack params into sub-configs
        project_config = ProjectConfig.from_dict(params["project"])
        plotter_config = PlotterConfig.from_dict(get_dict(params, "plotter"))
        sampler_config = SamplerConfig.from_dict(get_dict(params, "sampler"))
        problem_config = ProblemConfig.from_dict(problem_params)
        ml_config = MLConfig.from_dict(ml_params)
        nnfe_config = NNFEParamsConfig.from_dict(get_dict(params, "NNFE"))

        return cls(project=project_config,
                   plotter=plotter_config,
                   sampler=sampler_config,
                   FE=problem_config,
                   ML=ml_config,
                   NNFE=nnfe_config)