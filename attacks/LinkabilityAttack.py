import torch


class LinkabilityAttack:
    """
    Class for mounting linkability attack on models in Collaborative Learning.

    """

    def __init__(
        self,
        num_clients,
        client_datasets,
        loss,
        eval_batch_size=16,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.num_clients = num_clients
        self.client_datasets = client_datasets
        self.loss = loss
        self.eval_batch_size = eval_batch_size
        self.device = device

    def eval_loss(self, model, dataset):
        """
        Evaluate the loss on the training set

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        trainset = torch.utils.data.DataLoader(
            dataset, batch_size=self.eval_batch_size, shuffle=False
        )
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item() * self.eval_batch_size
                count += 1
        loss = epoch_loss / (count * self.eval_batch_size)
        # logging.debug("Loss after iteration: {}".format(loss))
        return loss

    def attack(self, model, skip=[]):
        """
        Function to mount linkability attack on the model.

        Args:
            model (torch.nn.Module): Model to be attacked.

        Returns:
            Attack results like linked identity

        """
        min_loss = 10e6
        predicted_client = None
        for client in range(self.num_clients):
            if client not in skip:
                cur_loss = self.eval_loss(model, self.client_datasets.use(client))
                if cur_loss < min_loss:
                    min_loss = cur_loss
                    predicted_client = client
        return predicted_client

    def log_all_losses(self, model, skip=[]):
        """
        Function to mount linkability attack on the model.

        Args:
            model (torch.nn.Module): Model to be attacked.

        Returns:
            Attack results like linked identity

        """
        losses = {}
        for client in range(self.num_clients):
            if client not in skip:
                cur_loss = self.eval_loss(model, self.client_datasets.use(client))
                losses[f"loss_trainset_{client}"] = cur_loss
        return losses


def main():
    pass
