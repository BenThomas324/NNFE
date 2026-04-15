"""
Frozen dataclasses representing the full NNFE configuration hierarchy.

The top-level :class:`NNFEConfig` is assembled from six sub-configs and is
normally loaded via :meth:`NNFEConfig.from_yaml`.  Both the ``FE`` and ``ML``
sections may be inlined or referenced as separate YAML files::

    FE: fe_problem.yaml   # path to a separate FE config file
    ML:                   # or inlined directly
      networks: ...

Config hierarchy::

    NNFEConfig
    ├── project:  ProjectConfig
    ├── plotter:  PlotterConfig
    ├── sampler:  SamplerConfig
    ├── FE:       cardiax.ProblemConfig
    ├── ML:       MLConfig
    └── NNFE:     NNFEParamsConfig
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml
from cardiax.input_file.config import ProblemConfig
from .ml_config import MLConfig
from .utils import validate_sampler, get_dict, get_Path


@dataclass(frozen=True)
class ProjectConfig:
    """Filesystem and run-management settings.

    Attributes:
        name: Human-readable project name; used as a directory prefix.
        parent_dir: Root directory under which the run folder is created.
        save: Whether to persist model weights and configuration to disk.
        print_progress: Print loss to stdout every *N* epochs.  ``None``
            disables printing.
        save_progress: Save model weights every *N* epochs during training.
            ``None`` disables intermediate saves.
        extra_dirs: Mapping of logical names (e.g. ``"model_dir"``,
            ``"plot_dir"``) to subdirectory paths relative to the run root.
        trained_weights_path: Optional path to pre-trained model weights
            loaded at startup (before training).
    """

    name: str = "Project"
    parent_dir: Path = Path(".")
    save: bool = True
    print_progress: int = None
    save_progress: int = None
    extra_dirs: dict = None
    trained_weights_path: str = None

    @classmethod
    def from_dict(cls, params: dict) -> "ProjectConfig":
        """Construct a :class:`ProjectConfig` from a raw config dictionary.

        Args:
            params: Dict with keys ``name``, ``parent_dir``, ``save``,
                ``print_progress``, ``save_progress``, and optionally
                ``extra_dirs`` and ``trained_weights_path``.

        Returns:
            A new :class:`ProjectConfig` instance.
        """
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


@dataclass(frozen=True)
class PlotterConfig:
    """Settings controlling which diagnostic plots are generated.

    Attributes:
        plot_loss: Whether to plot the training loss curve.
        plot_lr: Whether to plot the learning-rate schedule.
        plot_sample: Whether to plot the sampling distribution (currently
            unused but reserved for future use).
    """

    plot_loss: bool = True
    plot_lr: bool = True
    plot_sample: bool = True

    @classmethod
    def from_dict(cls, params: dict) -> "PlotterConfig":
        """Construct a :class:`PlotterConfig` from a raw config dictionary.

        Args:
            params: Dict with keys ``plot_loss`` and ``plot_lr``.

        Returns:
            A new :class:`PlotterConfig` instance.
        """
        return cls(plot_loss=params["plot_loss"],
                   plot_lr=params["plot_lr"])


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for the training and testing point samplers.

    Stores *what* to sample (sampler type and bounds); the actual sampling is
    performed by :class:`~nnfe.sampling.Sampler`.

    Attributes:
        training_sampler: Name of the sampler used for training points.
            Currently only ``"uniform"`` is supported.
        training_kwargs: Keyword arguments forwarded to the training sampler
            (e.g. ``mins``, ``maxes``, ``samples``).
        testing_sampler: Name of the sampler used for test points.
        testing_kwargs: Keyword arguments forwarded to the testing sampler.
    """

    training_sampler: str = "uniform"
    training_kwargs: dict = None
    testing_sampler: str = "uniform"
    testing_kwargs: dict = None

    @classmethod
    def from_dict(cls, params: dict) -> "SamplerConfig":
        """Construct a :class:`SamplerConfig` from a raw config dictionary.

        Args:
            params: Dict with keys ``training_sampler``, ``testing_sampler``,
                ``training_kwargs``, and ``testing_kwargs``.

        Returns:
            A new :class:`SamplerConfig` instance.

        Raises:
            ValueError: If either sampler name is not recognised.
        """
        training_sampler = validate_sampler(params["training_sampler"])
        testing_sampler = validate_sampler(params["testing_sampler"])

        return cls(training_sampler=training_sampler,
                   testing_sampler=testing_sampler,
                   training_kwargs=params["training_kwargs"],
                   testing_kwargs=params["testing_kwargs"])


@dataclass(frozen=True)
class NNFEParamsConfig:
    """NNFE-specific parameter configuration describing which FE quantities
    the network controls.

    The NNFE method parameterises FE internal variables and/or Dirichlet
    boundary conditions with the neural network output.  This config specifies
    *which* variables are parameterised and in what order they appear in the
    network output vector.

    Attributes:
        natural: Nested dict mapping FE keys to the internal variables
            (volumetric or surface) that are controlled by the natural
            (Neumann-type) outputs of the network.  Structure::

                natural:
                  internal:
                    <fe_key>:
                      <var_name>: ...
                  surface:
                    <fe_key>:
                      <bc_name>:
                        <var_name>: ...

        essential: Nested dict for Dirichlet (essential) boundary condition
            outputs (currently unused in the gradient computation but
            reserved).
        natural_order: Ordered list of variable names that maps network output
            indices to natural variables (used for slicing the output vector).
        essential_order: Ordered list of variable names for essential
            (Dirichlet) outputs.  An empty list disables Dirichlet training.
    """

    natural: dict = None
    essential: dict = None
    natural_order: list = None
    essential_order: list = None

    @classmethod
    def from_dict(cls, params: dict) -> "NNFEParamsConfig":
        """Construct an :class:`NNFEParamsConfig` from a raw config dictionary.

        Args:
            params: Dict with keys ``natural_order``, ``essential_order``,
                and optionally ``natural`` and ``essential``.

        Returns:
            A new :class:`NNFEParamsConfig` instance.
        """
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
    """Top-level configuration for the full NNFE solver.

    Aggregates all sub-configs required to instantiate
    :class:`~nnfe.nnfe_object.NNFE` via :meth:`NNFE.from_config`.

    Attributes:
        project: Filesystem and run-management settings.
        plotter: Diagnostic plot settings.
        sampler: Training/testing point sampler settings.
        FE: Finite element problem configuration (from ``cardiax``).
        ML: Neural network and optimizer configuration.
        NNFE: NNFE-specific parameter mapping configuration.
    """

    project: ProjectConfig
    plotter: PlotterConfig
    sampler: SamplerConfig
    FE: ProblemConfig
    ML: MLConfig
    NNFE: NNFEParamsConfig

    @classmethod
    def from_yaml(cls, path: Path) -> "NNFEConfig":
        """Load a complete :class:`NNFEConfig` from a YAML file.

        The ``FE`` and ``ML`` sections may either be inlined in the main YAML
        or provided as paths to separate ``.yaml`` files.

        Args:
            path: Path to the top-level NNFE config YAML file.

        Returns:
            A fully populated :class:`NNFEConfig` instance.
        """
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