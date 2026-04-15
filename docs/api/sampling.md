
# Sampling

NNFE is a data-free method: no ground-truth solution data is required.
`Sampler` generates control-variable point sets from a prescribed
distribution and provides mini-batch draws for the training loop.

## Sampler

::: nnfe.sampling.Sampler
    options:
        members:
            - uniform
            - draw_batch
            - safe_eval
