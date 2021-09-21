import copy
import sys
from datetime import datetime as dt
import os
from pathlib import Path
from threading import Event
from typing import Optional, Tuple
from datetime import datetime

import torch
import torch.utils.data
from tqdm import tqdm

from ai_based.data_handling.training_batch import TrainingBatch
from ai_based.utilities.evaluators import BaseEvaluator
from .training_session import TrainingSession
from . import CHECKPOINT_FILENAME


class Trainer:
    """
    Trainer class that manages the training of models. A single instance can be used to train multiple models using
    different hyper parameters. The class encapsulates the high level training logic, e.g. loading data, starting
    training sessions, logging, deciding when to stop training etc.
    """
    def __init__(self, config, training_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset,
                 checkpointing_enabled: bool, checkpointing_cyclic_epoch: Optional[int]):
        """
        Initialize a new Trainer object.
        :param config: Dictionary with parameters defining the general trainer behavior, e.g. verbosity, logging, etc.
        :type config: dict

        :param training_dataset: Dataset used for training
        :type training_dataset: torch.utils.data.DataSet

        :param test_dataset: Dataset used for validation
        :type test_dataset: torch.utils.data.DataSet
        """
        if not type(config) is dict:
            raise Exception("Error: The passed config object is not a dictionary.")
        self.config = config
        self.checkpointing_enabled = checkpointing_enabled
        self.checkpointing_cyclic_epoch = checkpointing_cyclic_epoch

        batch_size_test = config["batch_size_test"] if "batch_size_test" in config and config["batch_size_test"] is not None else config["batch_size"]
        self.data_loader_training = torch.utils.data.DataLoader(training_dataset, config["batch_size"], shuffle=True,
                                                                num_workers=config["num_loading_workers"],
                                                                collate_fn=TrainingBatch.from_iterable,
                                                                drop_last=True)
        self.data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size_test, shuffle=False,
                                                            num_workers=config["num_loading_workers"],
                                                            collate_fn=TrainingBatch.from_iterable)
        self.logged_batch_indices = self._calculate_logging_iterations()
        self.evaluator_type: type = config["evaluator_type"]

    def train(self, model, hyperparams, save_dir: Path = None):
        """
        Trains the passed model on the training set with the specified hyper-parameters.
        Loss, validation errors or other intermediate results are logged (or printed/plotted during the training) and
        saved to disk at the end.

        :param model: Model to train.
        :type model: torch.nn.Module

        :param hyperparams: Dictionary with all the hyperparameters required for training.
        :type hyperparams: dict

        :param save_dir: Directory where the results (weights, logs, eval results) are going to be stored at. If None,
                         nothing will be saved.


        :return: log: Dictionary containing all logs collected during training.
        :type: log: dict

        :return: final_validation_eval_results: The validation results (all metrics) of the model when using the best
                                                weights after training finished.
        :type: final_validation_eval_results: dict

        :return: best_weights: Model weights that performed best on the validation set during the whole training.
        :type: best_weights: dict
        """
        if save_dir is not None:
            save_dir = save_dir.resolve()
            assert save_dir.exists() and save_dir.is_dir(), f"Given save_dir '{save_dir}' not exists or is no folder"
        training_start_time = dt.now()
        print("Time: ", training_start_time.strftime("%H:%M:%S"))

        print()
        print("Setting things up...")
        self._print_hyperparameters(hyperparams, self.config["interest_keys"], indent=1)
        log_dict = self._initialize_log_dict()
        training_session = TrainingSession(model, evaluator_type=self.evaluator_type, hyperparams=hyperparams)

        print()
        print("\tChecking initial performance on test dataset:")
        started_at = datetime.now()
        best_evaluator_test = training_session.test_model(self.data_loader_test, dataset_type="test")
        best_evaluator_test.print_exhausting_metrics_results(indent=2)
        best_weights = copy.deepcopy(model.state_dict())
        print(f"\tThat took {(datetime.now() - started_at).total_seconds():.2f}s")

        print()
        print("All set, let's get started!", flush=True)
        for epoch_index in range(self.config["num_epochs"]):
            epoch_start_time = dt.now()
            running_training_loss = 0.0
            training_session.schedule_learning_rate()
            desc = f"Epoch {epoch_index+1}/{self.config['num_epochs']}"
            for i, batch in tqdm(enumerate(self.data_loader_training), desc=desc, total=len(self.data_loader_training), file=sys.stdout, position=0):
                training_loss, training_output = training_session.train_batch(batch)
                # aggregated_test_results = self._handle_logging(log_dict, training_session, training_loss, training_output, batch.ground_truth, i)
                running_training_loss += training_loss

            # Epoch finished
            average_training_loss = running_training_loss / len(self.data_loader_training)
            training_session.scheduler_metric = average_training_loss

            # if aggregated_test_results is None:
            evaluator_test = training_session.test_model(dataloader=self.data_loader_test, dataset_type="test")
            if self.config["determine_train_dataset_performance"] is True:
                evaluator_train = training_session.test_model(dataloader=self.data_loader_training, dataset_type="train")

            do_epoch_based_checkpointing = self.checkpointing_cyclic_epoch is not None and ((epoch_index+1) % self.checkpointing_cyclic_epoch) == 0
            if evaluator_test > best_evaluator_test or do_epoch_based_checkpointing:
                if do_epoch_based_checkpointing:
                    print("-> Cycle-based checkpointing")
                best_evaluator_test = evaluator_test
                best_weights = copy.deepcopy(model.state_dict())
                log_dict["best_epoch_index"] = epoch_index
                if self.checkpointing_enabled is True and save_dir is not None:
                    torch.save(best_weights, save_dir / CHECKPOINT_FILENAME)
                    print("-> Checkpoint saved")

            if log_dict["best_epoch_index"] is None:
                best_epoch_str = "/initial weights/"
            elif log_dict['best_epoch_index'] is not None:
                best_epoch_str = str(log_dict["best_epoch_index"] + 1)
                if log_dict['best_epoch_index'] == epoch_index:
                    best_epoch_str += " (this)"
            print(f"Epoch {epoch_index+1}/{self.config['num_epochs']}\n"
                  f"  - Epoch duration: {self._get_elapsed_time_str(epoch_start_time)}\n"
                  f"  - Average training loss: {average_training_loss}\n"
                  f"  - Best epoch: {best_epoch_str}")
            print(f"  - Validation results on test data:\n"
                  f"    + Short summary: {evaluator_test.get_short_summary()}\n"
                  f"    + {evaluator_test.get_scores_dict()}")
            if self.config["determine_train_dataset_performance"] is True:
                print(f"  - Validation results on training data:\n"
                      f"    + Short summary: {evaluator_train.get_short_summary()}\n"
                      f"    + {evaluator_train.get_scores_dict()}")

        # Training finished
        print()
        print("-" * 30)
        print(f"Training finished. Obtaining{' and saving' if save_dir else ''} results..")
        print("Final validation performance:")
        model.load_state_dict(best_weights)
        final_evaluator_test = training_session.test_model(self.data_loader_test, dataset_type="test")
        final_evaluator_test.print_exhausting_metrics_results(indent=1)
        print()

        if save_dir is not None:
            torch.save(log_dict, save_dir / "log.pt")
            torch.save(final_evaluator_test.get_scores_dict(), save_dir / "eval.pt")
            torch.save(best_weights, save_dir / "weights.pt")

        return log_dict, final_evaluator_test.get_scores_dict(), best_weights

    def _calculate_logging_iterations(self):
        max_idx = len(self.data_loader_training) - 1
        interval = len(self.data_loader_training) // self.config["logging_frequency"]
        logging_iterations = [max_idx - log * interval for log in range(0, self.config["logging_frequency"])]
        return logging_iterations

    def _initialize_log_dict(self):
        evaluator_ = self.evaluator_type.empty()
        scores_template = {score: [] for score in evaluator_.get_scores_dict().keys()}
        log = {"training": copy.deepcopy(scores_template),
               "test": copy.deepcopy(scores_template)}
        log["training"]["loss"] = []
        log["training"]["grad"] = []
        log["best_epoch_index"] = None
        return log

    def _handle_logging(self, log_dict, training_session, training_loss, training_output, training_ground_truth, batch_index):
        if self.config["log_loss"]:
            log_dict["training"]["loss"].append(training_loss)

        if self.config["log_grad"]:
            log_dict["training"]["grad"].append(self._sum_gradients(training_session.model))

        aggregated_test_results = None
        if batch_index in self.logged_batch_indices:
            training_eval_results = self.evaluator(training_output, training_ground_truth)
            aggregated_test_results = training_session.test_model(self.data_loader_test, dataset_type="test")

            for score_name, value in training_eval_results.items():
                log_dict["training"][score_name].append(value)

            for score_name, value in aggregated_test_results.items():
                log_dict["test"][score_name].append(value)

            if self.config["verbose"]:
                print(f"Iteration {batch_index}/{len(self.data_loader_training)}: Loss: {training_loss:.2f}")
                print("\tTraining  : ", end="")
                self.evaluator.print_exhausting_metrics_results(training_eval_results, flat=True)
                print("\tValidation: ", end="")
                self.evaluator.print_exhausting_metrics_results(aggregated_test_results, flat=True)
                print()
        return aggregated_test_results

    @staticmethod
    def _print_hyperparameters(hyperparams, keys, indent=0):
        print("\t" * indent + "The hyper-parameter interest keys are:")
        for key in keys:
            val = hyperparams
            for sub_key in key:
                val = val[sub_key]

            print("\t" * indent + "\t{}: \t{}".format(key, val))

    @staticmethod
    def _get_elapsed_time_str(start_time) -> str:
        time_elapsed = dt.now() - start_time
        hours, remainder = divmod(time_elapsed.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return "{:.0f}h {:.0f}m {:.0f}s".format(hours, minutes, seconds)

    @staticmethod
    def _sum_gradients(model):
        grad_sum = 0.0
        for p in model.parameters():
            grad_sum += p.grad.abs().sum()
        return grad_sum
