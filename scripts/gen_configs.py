import argparse

from arcsf.config.experiment import generate_experiment_configs


def main():
    parser = argparse.ArgumentParser(
        description=("Generates experiment configs for the top level experiment.")
    )
    parser.add_argument(
        "experiment_name",
        type=str,
        help="Name of experiment yaml file contained in configs/experiment",
    )
    args = parser.parse_args()
    generate_experiment_configs(args.experiment_name)


if __name__ == "__main__":
    main()
