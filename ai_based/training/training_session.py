from datetime import datetime

import torch


class TrainingSession:
    """
    Class for a single training session for a specific model with fixed hyper-parameters.
    The class encapsulates low level training logic associated with the model and optimizer i.e.
    running the model"s forward/backward path, updating model parameters, run model in test mode and
    scheduling the learning rate.
    """
    def __init__(self, model, evaluator, hyperparams):
        if not type(hyperparams) is dict:
            raise Exception("Error: The passed config object is not a dictionary.")
        self.params = hyperparams
        self.model = model
        self.evaluator = evaluator
        self.device = model.device

        self.loss_function = hyperparams["loss_function"].to(model.device)
        self.optimizer = hyperparams["optimizer"](model.parameters(), **(hyperparams["optimizer_args"]))

        self.scheduler = hyperparams["scheduler"](self.optimizer, **(hyperparams["scheduler_args"]))
        if self.params["scheduler_requires_metric"]:
            self.scheduler_metric = float("inf")

    def test_model(self, dataloader):
        batchwise_results = []
        self.model.eval()
        with torch.no_grad():
            started_at = datetime.now()
            for batch in dataloader:
                batch.to_device(self.device)
                net_input = torch.autograd.Variable(batch.input_data)
                net_output = self.model(net_input)
                batchwise_results.append(self.evaluator(net_output, batch.ground_truth))
            print(f"Testing model took {(datetime.now()-started_at).total_seconds():.1f}s")
        self.model.train()
        aggregated_batch_results = self.evaluator.aggregate_batch_results(batchwise_results)
        return aggregated_batch_results

    def schedule_learning_rate(self):
        if self.params["scheduler_requires_metric"]:
            self.scheduler.step(self.scheduler_metric)
        else:
            self.scheduler.step()

    def train_batch(self, batch):
        batch.to_device(self.device)
        self.optimizer.zero_grad()

        net_input = torch.autograd.Variable(batch.input_data)
        net_output = self.model(net_input)

        loss = self._backward_and_optimize(net_output, batch.ground_truth)

        return loss, net_output

    def _backward_and_optimize(self, net_output, ground_truth):
        # print(net_output.shape, net_output)
        # print(ground_truth.shape, ground_truth)
        loss = self.loss_function(net_output, ground_truth)
        loss.backward()
        self.optimizer.step()
        loss = loss.data.cpu()
        return loss
