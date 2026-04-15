"""
Filesystem management for NNFE runs.

:class:`ProjectManager` is responsible for:

- Creating the run directory tree under ``parent_dir / name / <rng_key>``.
- Generating a unique integer RNG seed (used also as the directory name so
  each run is isolated and reproducible).
- Writing a sentinel file ``running.txt`` on startup (overwritten with
  elapsed-time on completion).
- Providing a ``paths`` dict that maps logical names (e.g. ``"model_dir"``)
  to resolved :class:`pathlib.Path` objects.
"""

import numpy as onp
from pathlib import Path


class ProjectManager:
    """Manages the on-disk run directory and associated metadata.

    Args:
        config: :class:`~nnfe.nnfe_config.ProjectConfig` instance.  If
            ``config.save`` is ``True``, a unique subdirectory is created
            under ``config.parent_dir / config.name`` and all extra
            directories listed in ``config.extra_dirs`` are created inside it.

    Attributes:
        config: :class:`~nnfe.nnfe_config.ProjectConfig` used to build this
            manager.
        save: Whether saving is enabled.
        parent: Absolute path to the unique run directory, or ``None`` if
            saving is disabled.
        key: Integer RNG seed derived from the directory name, or ``None``
            if saving is disabled.
        paths: Mapping of logical directory names to resolved
            :class:`pathlib.Path` objects (populated from
            ``config.extra_dirs``).
        print: Print-progress interval (epochs), or ``False`` if disabled.
        save_progress: Intermediate-save interval (epochs), or ``False`` if
            disabled.
        trained_weights_path: Path to pre-trained weights, if specified in
            config.
    """

    def __init__(self, config):
        self.config = config
        self.save = config.save if config.save is not None else False

        if config.save:
            self.parent, self.key = self.generate_parent_and_rng()
            self.parent.mkdir(parents=True, exist_ok=False)
            self.paths = self.create_dirs()
        else:
            self.parent = None
            self.key = None
            self.paths = {}

        self.print = config.print_progress if config.print_progress is not None else False
        self.save_progress = config.save_progress if config.save_progress is not None else False

        if config.trained_weights_path is not None:
            self.trained_weights_path = config.trained_weights_path

        return

    def generate_parent_and_rng(self):
        """Generate a unique run directory path and integer RNG seed.

        Picks a random integer in ``[0, 1e5)`` and checks that a
        correspondingly named subdirectory does not already exist, retrying
        until a free name is found.

        Returns:
            Tuple of ``(parent_path, key)`` where *parent_path* is a
            :class:`pathlib.Path` and *key* is the integer seed.
        """
        parent = Path(self.config.parent_dir) / Path(self.config.name)

        temp_key = onp.random.randint(1e5)
        while (parent / f"{temp_key}").exists():
            temp_key = onp.random.randint(1e5)
        return parent / Path(f"{temp_key}"), temp_key

    def create_dirs(self):
        """Create all subdirectories listed in ``config.extra_dirs``.

        Also writes a ``running.txt`` sentinel file (value ``1``) so
        monitoring scripts can detect that a run is in progress.

        Returns:
            Dict mapping each key in ``config.extra_dirs`` to its resolved
            :class:`pathlib.Path`.
        """
        paths = {}
        for dir_key, dir_path in self.config.extra_dirs.items():
            dir_path = self.parent / dir_path
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
            paths[dir_key] = dir_path
        onp.savetxt(self.parent / "running.txt", onp.array([1]))
        return paths

