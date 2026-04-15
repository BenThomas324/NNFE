
# Project Manager

`ProjectManager` handles all filesystem operations for a run: creating
the unique run directory, generating an integer RNG seed, creating
subdirectories, and writing the sentinel `running.txt` file.

## ProjectManager

::: nnfe.project_manager.ProjectManager
    options:
        members:
            - generate_parent_and_rng
            - create_dirs
