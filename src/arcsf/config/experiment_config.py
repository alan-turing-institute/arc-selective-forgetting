import os
import warnings
from itertools import product

import yaml
from jinja2 import Environment, FileSystemLoader

from arcsf.config.config_class import Config
from arcsf.models.config_class import ModelConfig


def _listify(obj: object):
    if isinstance(obj, list):
        return obj
    return [obj]


def generate_experiment_configs(
    top_config_name: str,
) -> None:

    # Read in yaml file
    with open(f"configs/experiment/{top_config_name}.yaml") as f:
        top_config = yaml.safe_load(f)

    # Loop over, construct combo dict
    combo_dict = {}
    for key, value in top_config["configs"].items():
        combo_dict[key] = _listify(value)

    # Construct full set of combinations
    sweep_dict_keys, sweep_dict_vals = zip(*combo_dict.items())
    combo_dicts = [
        dict(zip(sweep_dict_keys, v)) for v in product(*list(sweep_dict_vals))
    ]

    # Check this is a reasonably number of jobs for an array of training jobs
    if len(combo_dicts) > 1001:
        warnings.warn("Slurm array jobs cannot exceed more than 1001!")

    # Check whether to generate baskerville scripts:
    use_bask = False
    if top_config["use_bask"]:
        use_bask = True
        if not os.path.exists("bask"):
            os.mkdir("bask")
        baskdir = f"bask/{top_config_name}"
        if not os.path.exists(baskdir):
            os.mkdir(baskdir)

    # Write out dicts and optionally bask scripts
    outdir = f"configs/experiment/{top_config_name}"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for n, combo in enumerate(combo_dicts):
        file_name = f"{outdir}/run_{n}.yaml"
        with open(file_name, "w") as f:
            yaml.dump(combo, f)
        if use_bask:
            environment = Environment(loader=FileSystemLoader("src/arcsf/config"))
            template = environment.get_template("jobscript_template.sh")

            script_content = template.render(
                job_name=f"submit_{n}",
                walltime=top_config["bask"]["walltime"],
                node_number=top_config["bask"]["node_number"],
                gpu_number=top_config["bask"]["gpu_number"],
                cpu_per_gpu=top_config["bask"]["cpu_per_gpu"],
                # array_number=, #TODO
                script_name="scripts/train.py",
                experiment_file=file_name,
            )

            with open(f"{baskdir}/submit_{n}.sh", "w") as f:
                f.write(script_content)


class ExperimentConfig(Config):

    def __init__(
        self,
        model_yaml_name: str | None,
    ) -> None:
        super().__init__()

        # Load in other configs
        self.data_config = None  # TODO: replace this
        self.model_config = ModelConfig.from_yaml(model_yaml_name)

    @classmethod
    def from_dict(cls, dict) -> "ModelConfig":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_id", "model_kwargs", and
                "trainer_kwargs" keys.

        Returns:
            FineTuningConfig object.
        """
        return cls(**dict)

    # TODO: define to_dict method
