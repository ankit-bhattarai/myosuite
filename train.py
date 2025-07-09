import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call
from jax import numpy as jnp

def print_config(args: DictConfig):
    parsed_args = OmegaConf.to_yaml(OmegaConf.to_container(args, resolve=True))
    print(parsed_args)

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(args: DictConfig):
    parsed_args = OmegaConf.to_yaml(OmegaConf.to_container(args, resolve=True))
    # print(parsed_args)
    activation = instantiate(args.network.activation, _partial_=True)
    # print(activation)
    network = args.network.network_factory
    # print(network)
    print_config(args.env)
    print_config(args.renderer)
    print_config(args)
if __name__ == "__main__":
    main()