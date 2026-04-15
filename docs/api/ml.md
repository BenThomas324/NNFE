
# Machine Learning Manager

`MLManager` owns the neural network, optimizer, and training state.  It is
constructed automatically by [`NNFE.from_config`][nnfe.nnfe_object.NNFE.from_config]
but can also be used standalone for generic JAX/Equinox training workflows.

## MLManager

::: nnfe.ml.MLManager
    options:
        members:
            - from_config
            - from_yaml
            - create_network
            - create_optimizer
            - network_from_config
            - init_linear_weight
            - load_network
            - filtering
            - trunc_weight
            - trunc_bias
            - dump_config
