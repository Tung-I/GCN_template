import torch
import dgl
from tqdm import tqdm
from src.runner.trainers.base_trainer import BaseTrainer


class KitsClfTrainer(BaseTrainer):
    """The KiTS trainer for segmentation task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def _run_epoch(self, mode):
        """Run an epoch for training.
        Args:
            mode (str): The mode of running an epoch ('training' or 'validation').
        Returns:
            log (dict): The log information.
            batch (dict or sequence): The last batch of the data.
            outputs (torch.Tensor or sequence of torch.Tensor): The corresponding model outputs.
        """
        if mode == 'training':
            self.net.train()
        else:
            self.net.eval()
        dataloader = self.train_dataloader if mode == 'training' else self.valid_dataloader
        trange = tqdm(dataloader,
                      total=len(dataloader),
                      desc=mode)

        log = self._init_log()

        for m in self.metric_fns:
            m.reset()

        for batch in trange:
            batch = self._allocate_data(batch)
            graphs, labels = self._get_inputs_targets(batch)
            if mode == 'training':
                outputs = self.net(graphs)
                losses = self._compute_losses(outputs, labels)
                loss = (torch.stack(losses) * self.loss_weights).sum()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    outputs = self.net(graphs)
                    losses = self._compute_losses(outputs, labels)
                    loss = (torch.stack(losses) * self.loss_weights).sum()
            metrics =  self._compute_metrics(outputs, labels)
            batch_size = self.train_dataloader.batch_size if mode == 'training' else self.valid_dataloader.batch_size
            self._update_log(log, batch_size, loss, losses, metrics)
            trange.set_postfix(**dict((key, f'{value: .3f}') for key, value in log.items()))

        return log, batch, outputs


    def _get_inputs_targets(self, batch):
        """Specify the data input and target.
        Args:
            batch (dict): A batch of data.
        Returns:
            input (torch.Tensor): The data input.
            target (torch.LongTensor): The data target.
        """
        features = batch['feature']
        labels = batch['label']
        adj_arrs = batch['adj_arr']

        n_batch = labels.size(0)

        graphs = []
        for i in range(n_batch):
            feature = features[i]
            label = labels[i]
            adj_arr = adj_arrs[i]

            n_node = feature.size(0)
            g = dgl.DGLGraph()
            g.add_nodes(n_node)
            for i in tqdm(range(n_node)):
                src = list(range(n_node))
                dst = [i] * n_node
                g.add_edges(src, dst)
                val = torch.FloatTensor(n_node, 2)
                val[:, 0] = adj_arr[i]
                val[:, 1] = adj_arr[i]
                g.edges[src, dst].data['w'] = val 
            g.ndata['x'] = feature
            g.to(self.device)
            # print(g.ndata['x'].device)
            graphs.append(g)

        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels

    def _compute_losses(self, output, target):
        """Compute the losses.
        Args:
            output (torch.Tensor): The model output.
            target (torch.LongTensor): The data target.
        Returns:
            losses (list of torch.Tensor): The computed losses.
        """
        losses = [loss(output, target) for loss in self.loss_fns]
        return losses

    def _compute_metrics(self, output, target):
        """Compute the metrics.
        Args:
             output (torch.Tensor): The model output.
             target (torch.LongTensor): The data target.
        Returns:
            metrics (list of torch.Tensor): The computed metrics.
        """
        metrics = [metric(output, target) for metric in self.metric_fns]
        return metrics

    def _init_log(self):
        """Initialize the log.
        Returns:
            log (dict): The initialized log.
        """
        log = {}
        log['Loss'] = 0
        for loss in self.loss_fns:
            log[loss.__class__.__name__] = 0
        for metric in self.metric_fns:
            log[metric.__class__.__name__] = 0
        return log

    def _update_log(self, log, batch_size, loss, losses, metrics):
        """Update the log.
        Args:
            log (dict): The log to be updated.
            batch_size (int): The batch size.
            loss (torch.Tensor): The weighted sum of the computed losses.
            losses (list of torch.Tensor): The computed losses.
            metrics (list of torch.Tensor): The computed metrics.
        """
        log['Loss'] = loss.item()
        for loss, _loss in zip(self.loss_fns, losses):
            log[loss.__class__.__name__] = _loss.item()
        for metric, _metric in zip(self.metric_fns, metrics):
            log[metric.__class__.__name__] = _metric.item()

        # log['Loss'] += loss.item() * batch_size
        # for loss, _loss in zip(self.loss_fns, losses):
        #     log[loss.__class__.__name__] += _loss.item() * batch_size
        # for metric, _metric in zip(self.metric_fns, metrics):
        #     if metric.__class__.__name__ == 'Dice':
        #         log['Dice'] += _metric.mean().item() * batch_size
        #         for i, class_score in enumerate(_metric):
        #             log[f'Dice_{i}'] += class_score.item() * batch_size
        #     else:
        #         log[metric.__class__.__name__] += _metric.item() * batch_size
