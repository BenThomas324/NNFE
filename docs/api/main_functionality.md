
# Main Functionalities

## NNFE

The top-level solver object.  Construct it from a YAML config file with
[`NNFE.from_yaml`][nnfe.nnfe_object.NNFE.from_yaml] or from an
[`NNFEConfig`][nnfe.nnfe_config.NNFEConfig] dataclass with
[`NNFE.from_config`][nnfe.nnfe_object.NNFE.from_config].

::: nnfe.nnfe_object.NNFE
    options:
        members:
            - from_yaml
            - from_config

## Train

::: nnfe.nnfe_object.NNFE.train

## Test

::: nnfe.nnfe_object.NNFE.test

## Evaluate

::: nnfe.nnfe_object.NNFE.evaluate

## Save

::: nnfe.nnfe_object.NNFE.save

## Dump Config

::: nnfe.nnfe_object.NNFE.dump_config