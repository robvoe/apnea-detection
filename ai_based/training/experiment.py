import shutil
import os
import copy
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

import ai_based.data_handling.ai_datasets
from util.paths import RESULTS_PATH_AI
from .trainer import Trainer
from . import CHECKPOINT_FILENAME


class Experiment:
    """
    In a single experiment an arbitrary number of models can be trained sequentially using different configurations in
    a grid search approach.
    1. Configure experimental setup by setting base parameters and add hyperparameter options using `add_option()`.
    2. Start the whole process using the `run()` method.
    Only a single dataset per experiment is supported.
    The results are stored to the "results" directory. The class creates a new subfolder (experiment directory) equal to
    the experiment name (specified in the config). Under this directory it creates a results directory for each distinct
    hyperparamter combination (using increasing integers as names) and adds another layer of repetition subdirectories
    for each repetition.
    The information that is saved:
      - config.pt: experiment config and trainer config (placed only once in experiment directory)
      - params.pt: hyperparameters in each result directory
      - log.pt: training log for a single training session (loss, intermediate evaluations)
      - eval.pt: final results (all metrics)
      - weights.pt: final weights of the model (from the best epoch)
    """
    SILENTLY_OVERWRITE_PREVIOUS_RESULTS = False

    def __init__(self, experiment_config, trainer_config, base_hyperparams):
        self.config = {
            "experiment": experiment_config,
            "trainer": trainer_config,
        }
        self.base_hyperparams = base_hyperparams
        self.options = {}

    @classmethod
    def _purge_folder_recursively_excluding_checkpoints(cls, folder: Path):
        """Deletes all sub-contents of a folder. Checkpoint files will not be removed."""
        if not folder.exists() or not folder.is_dir():
            return
        for subpath_ in folder.iterdir():
            if subpath_.is_dir():
                cls._purge_folder_recursively_excluding_checkpoints(subpath_)
                if len(list(subpath_.iterdir())) == 0:
                    subpath_.rmdir()
            elif subpath_.name.lower() != CHECKPOINT_FILENAME.lower():
                subpath_.unlink()
        pass

    def add_options(self, param_name, values):
        """
        Register a certain parameter by its name and specify a list of values that should be tried later in a grid
        search. A call to that function increases the total number of training sessions exponentially.
        :param param_name: Iterable of strings representing the keys (top-down) for the nested hyperparameter dictionary
        :param values: Iterable of values from the respective value domain of the parameter to be used.
        """
        self.options[param_name] = values
        self.config["trainer"]["interest_keys"].append(param_name)

    def run(self):
        experiment_dir = RESULTS_PATH_AI / self.config["experiment"]["name"]
        if experiment_dir.exists():
            assert experiment_dir.is_dir()
            if self.config["experiment"]["checkpointing_enabled"] is True:
                self._purge_folder_recursively_excluding_checkpoints(experiment_dir)
            elif self.config["experiment"]["checkpointing_enabled"] is False and len(list(experiment_dir.iterdir())):
                if not self.SILENTLY_OVERWRITE_PREVIOUS_RESULTS:
                    raise FileExistsError(f"Experiment directory '{experiment_dir}' is not empty. Remove before retry!")
                else:
                    shutil.rmtree(experiment_dir)
        os.makedirs(experiment_dir, exist_ok=True)
        torch.save(self.config, os.path.join(experiment_dir, "config.pt"))

        combinations_of_configs = self._generate_combinations()
        print(f"There are {len(combinations_of_configs)} parameter combinations we will go through.")

        # Try loading model weights prior to instantiating the datasets. Helps a lot finding erroneous weights files
        model = self._create_model(combinations_of_configs[0], checkpoint_lookup_dir=None)
        print(f"Number of model parameters = {model.num_parameters():,}")
        del model

        # Now let's load the data
        dataset_type = self.base_hyperparams["dataset_type"]
        if dataset_type is not None:
            if not issubclass(dataset_type, ai_based.data_handling.ai_datasets.BaseAiDataset):
                raise TypeError(f"Invalid dataset class specified: {dataset_type}")
            training_dataset = dataset_type(config=self.base_hyperparams["train_dataset_config"])
            test_dataset = dataset_type(config=self.base_hyperparams["test_dataset_config"])
            print("Successfully instantiated train & test datasets")
        elif dataset_type is None:
            assert all(_ in self.base_hyperparams for _ in ("train_dataset", "test_dataset"))
            training_dataset = self.base_hyperparams["train_dataset"]
            test_dataset = self.base_hyperparams["test_dataset"]
            assert training_dataset is not None and test_dataset is not None
            print("Using pre-instantiated train & test datasets")
        else:
            raise RuntimeError("We should never reach this point")
        self.config["experiment"]["train_set_size"] = len(training_dataset)
        self.config["experiment"]["test_set_size"] = len(test_dataset)
        print(f"train_dataset_size = {len(training_dataset):,}")
        print(f"test_dataset_size = {len(test_dataset):,}")

        trainer = Trainer(self.config["trainer"], training_dataset=training_dataset, test_dataset=test_dataset,
                          checkpointing_enabled=self.config["experiment"]["checkpointing_enabled"],
                          checkpointing_cyclic_epoch=self.config["experiment"]["checkpointing_cyclic_epoch"])

        experiment_started_at = datetime.now()
        for combination_index, hyperparams in enumerate(combinations_of_configs):
            print("\n\n" + "#" * 100)
            print("START OF SESSION {}/{}".format(combination_index + 1, len(combinations_of_configs)))

            combination_dir = Path(experiment_dir) / f"combination_{combination_index}"
            combination_dir.mkdir(parents=False, exist_ok=True)
            torch.save(hyperparams, combination_dir / "params.pt")

            for repetition_index in range(self.config["experiment"]["n_repetitions"]):
                print("\nRepetition {}/{}  ({}):".format(repetition_index + 1,
                                                         self.config["experiment"]["n_repetitions"],
                                                         self.config["experiment"]["name"]))
                print("*" * 50)

                if self.config["experiment"]["deterministic_mode"]:
                    torch.manual_seed(0)

                repetition_dir = combination_dir / f"repetition_{repetition_index}"
                repetition_dir.mkdir(parents=False, exist_ok=True)
                model = self._create_model(hyperparams, checkpoint_lookup_dir=repetition_dir)
                trainer.train(model, hyperparams, save_dir=repetition_dir)

        print(f"\nExperiment >> {self.config['experiment']['name']} << finished.")
        Experiment._final_output(experiment_started_at, combinations_of_configs, experiment_dir)

    @staticmethod
    def _final_output(experiment_started_at: datetime, combinations_of_configs, experiment_dir: str):
        """Outputs some final information to the screen and the file 'total_duration.txt'"""
        duration = datetime.now() - experiment_started_at
        duration_str = f"{duration.days}d, {duration.seconds // 3600}h:{(duration.seconds // 60) % 60}min"
        print(f"In total, all {len(combinations_of_configs)} trainings took {duration_str}.")

        with open(os.path.join(experiment_dir, "total_duration.txt"), mode="w") as file:
            file.write("\n".join([
                f"Total duration of experiment: {duration_str}",
                f"Total seconds of experiment: {int(duration.total_seconds())}",
                f"Trained combinations in total: {len(combinations_of_configs)}",
                f"Time of training start: {experiment_started_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Time of training end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]))

    def _generate_combinations(self):
        hyper_param_configs = [self.base_hyperparams]
        for param_name, values in self.options.items():
            new_list = []
            for i, config in enumerate(hyper_param_configs):
                for val in values:
                    new_config = copy.deepcopy(config)
                    sub_dict = new_config
                    for key in param_name[:-1]:
                        sub_dict = sub_dict[key]
                    sub_dict[param_name[-1]] = val
                    new_list.append(new_config)
            hyper_param_configs = new_list
        return hyper_param_configs

    def _create_model(self, hyperparams, checkpoint_lookup_dir: Optional[Path]):
        model_config = hyperparams["model_config"]
        model = hyperparams["model"](model_config)
        model.to(self.config["experiment"]["target_device"])

        # Load model weights
        assert not (self.config["experiment"]["init_weights_path"] is not None and self.config["experiment"]["checkpointing_enabled"] is True), \
            "Inconsistent parametrization. Choose only one of both 'init_weights_path' or 'checkpointing_enabled'!"

        if self.config["experiment"]["init_weights_path"] is not None:
            weights = torch.load(self.config["experiment"]["init_weights_path"],
                                 map_location=self.config["experiment"]["target_device"])
            model.load_state_dict(weights)
        elif self.config["experiment"]["checkpointing_enabled"] is True and checkpoint_lookup_dir is not None:
            assert checkpoint_lookup_dir.exists() and checkpoint_lookup_dir.is_dir()
            checkpoint_file = checkpoint_lookup_dir / CHECKPOINT_FILENAME
            if checkpoint_file.exists() and checkpoint_file.is_file():
                weights = torch.load(checkpoint_file, map_location=self.config["experiment"]["target_device"])
                model.load_state_dict(weights)
                print("-> Successfully loaded checkpoint file")
        return model
