import sys
from datetime import datetime
from typing import Dict

import torch
from tqdm import tqdm

from ai_based.utilities.evaluators import BaseEvaluator
from ai_based.data_handling.training_batch import BaseTrainingBatch


class TrainingSession:
    """
    Class for a single training session for a specific model with fixed hyper-parameters.
    The class encapsulates low level training logic associated with the model and optimizer i.e.
    running the model"s forward/backward path, updating model parameters, run model in test mode and
    scheduling the learning rate.
    """
    def __init__(self, model, evaluator_type: type, hyperparams: Dict):
        if not type(hyperparams) is dict:
            raise Exception("Error: The passed config object is not a dictionary.")
        self.params = hyperparams
        self.model = model
        self.evaluator_type = evaluator_type
        self.device = model.device

        self.loss_function = hyperparams["loss_function"].to(model.device)
        self.optimizer = hyperparams["optimizer"](model.parameters(), **(hyperparams["optimizer_args"]))

        self.scheduler = hyperparams["scheduler"](self.optimizer, **(hyperparams["scheduler_args"]))
        if self.params["scheduler_requires_metric"]:
            self.scheduler_metric = float("inf")

    def test_model(self, dataloader, dataset_type: str) -> BaseEvaluator:
        assert dataset_type in ("train", "test")
        overall_evaluator = self.evaluator_type.empty()
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Testing model on {dataset_type} dataset", total=len(dataloader), leave=False, file=sys.stdout, position=0):
                batch.to_device(self.device)
                net_input = torch.autograd.Variable(batch.input_data)
                net_output = self.model(net_input)
                batch_evaluator = self.evaluator_type(model_output_batch=net_output, ground_truth_batch=batch.ground_truth)
                overall_evaluator += batch_evaluator
        self.model.train()
        return overall_evaluator

    def schedule_learning_rate(self):
        if self.params["scheduler_requires_metric"]:
            self.scheduler.step(self.scheduler_metric)
        else:
            self.scheduler.step()

    def train_batch(self, training_batch: BaseTrainingBatch):
        # Perform a few sanity checks. This helps debugging, totally!
        assert not torch.any(torch.isnan(training_batch.input_data))
        assert not torch.any(torch.isinf(training_batch.input_data))
        assert not torch.any(torch.isnan(training_batch.ground_truth))
        assert not torch.any(torch.isinf(training_batch.ground_truth))

        training_batch.to_device(self.device)
        self.optimizer.zero_grad()

        net_input = torch.autograd.Variable(training_batch.input_data)
        net_output = self.model(net_input)

        loss = self._backward_and_optimize(net_output, training_batch.ground_truth)

        if torch.isnan(loss):
            print(net_input.shape)
            print(net_output.shape)
            print(training_batch.ground_truth.shape)
            print(torch.any(torch.isnan(net_input)))
            print(torch.any(torch.isnan(net_output)))
            print(torch.any(torch.isnan(training_batch.ground_truth)))
            print(torch.any(torch.isinf(net_input)))
            print(torch.any(torch.isinf(net_output)))
            print(torch.any(torch.isinf(training_batch.ground_truth)))
        assert not torch.any(torch.isnan(loss))
        assert not torch.any(torch.isinf(loss))
        return loss, net_output

    def _backward_and_optimize(self, net_output, ground_truth):
        # print(net_output.shape, net_output)
        # print(ground_truth.shape, ground_truth)
        loss = self.loss_function(net_output, ground_truth)
        loss.backward()
        self.optimizer.step()
        loss = loss.data.cpu()
        return loss
