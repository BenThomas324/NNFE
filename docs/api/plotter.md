
# Plotter

`Plotter` generates diagnostic plots (loss curve, learning-rate schedule)
during and after training.  All figures are saved as PNG files so they work
in headless / HPC environments.

## Plotter

::: nnfe.plotter.Plotter
    options:
        members:
            - plot_loss
            - plot_learning_rate
            - plot_grad
