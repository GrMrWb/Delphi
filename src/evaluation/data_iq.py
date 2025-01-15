"""
Needs to be changed for a better understanding for the aleatoric and epistemic and for visualisation purpose


This was clone from https://github.com/vanderschaarlab/Data-IQ/ under the MIT license

Date: 16 May 2023
"""
from autograd_lib import autograd_lib
from src.delphi.strategy import *
from src.evaluation.utils import EvaluationMetrices
import logging, copy

logger = logging.getLogger(__name__)

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class DataIQ_Torch:
    def __init__(self, X, y, sparse_labels: bool = False, evaluation: bool = False, client ='Server'):
        """
        The function takes in the training data and the labels, and stores them in the class variables X
        and y. It also stores the boolean value of sparse_labels in the class variable _sparse_labels

        Args:
          X: the input data
          y: the true labels
          sparse_labels (bool): boolean to identify if labels are one-hot encoded or not. If not=True.
        Defaults to False
        """
        self.X = X
        self.y = y
        self._sparse_labels = sparse_labels
        self.client = client

        # placeholder
        self._gold_labels_probabilities = None
        self._true_probabilities = None
        self._adv_gold_labels_probabilities = None
        self._adv_true_probabilities = None
        self._grads = None
        
        if evaluation:
            self._accuracy, self._recall, self._precision = 0 , 0,  0
            self._adv_accuracy, self._adv_recall, self._adv_precision = 0 , 0,  0
            self.evaluation = EvaluationMetrices()
        else:
            self._accuracy, self._recall, self._precision = 0 , 0,  0
            self._adv_accuracy, self._adv_recall, self._adv_precision = 0 , 0,  0
            self.evaluation = EvaluationMetrices()
            

    def gradient(self, net, device='cpu'):
        """
        Used to compute the norm of the gradient through training

        Args:
          net: pytorch neural network
          device: device to run the computation on
        """

        # setup
        try:
            data = torch.tensor(self.X, device=device)
            targets = torch.tensor(self.y, device=device).long()
            loss_fn = torch.nn.CrossEntropyLoss()

            model = net.to(device)

            # register the model for autograd
            autograd_lib.register(model)

            activations = {}

            def save_activations(layer, A, _):
                activations[layer] = A

            with autograd_lib.module_hook(save_activations):
                output = model(data)
                loss = loss_fn(output, targets)

            norms = [torch.zeros(data.shape[0], device=device)]

            def per_example_norms(layer, _, B):
                A = activations[layer]
                try:
                    norms[0] += (A * A).sum(dim=1) * (B * B).sum(dim=1)
                except:
                    x = (A * A).sum(dim=1) * (B * B).sum(dim=1)
                    norms[0] += x.reshape(x.shape[0])

            with autograd_lib.module_hook(per_example_norms):
                loss.backward()

            grads_train = norms[0].cpu().numpy()

            if self._grads is None:  # Happens only on first iteration
                self._grads = np.expand_dims(grads_train, axis=-1)
            else:
                stack = [self._grads, np.expand_dims(grads_train, axis=-1)]
                self._grads = np.hstack(stack)
        except:
            data.requires_grad = True
            output = model(data)
            loss = loss_fn(output, targets)
            loss.backward()
    
    def on_epoch_end(self, net, device="cpu", **kwargs):
        """
        The function computes the gold label and true label probabilities over all samples in the
        dataset

        We iterate through the dataset, and for each sample, we compute the gold label probability (i.e.
        the actual ground truth label) and the true label probability (i.e. the predicted label).

        We then append these probabilities to the `_gold_labels_probabilities` and `_true_probabilities`
        lists.

        We do this for every sample in the dataset, and for every epoch.

        Args:
          net: the neural network
          device: the device to use for the computation. Defaults to cpu
        """

        # compute the gradient norm
        # self.gradient(net, device)

        # Compute both the gold label and true label probabilities over all samples in the dataset
        gold_label_probabilities = (    
            list()
        )  # gold label probabilities, i.e. actual ground truth label
        true_probabilities = list()  # true label probabilities, i.e. predicted label

        net = net.to(device)
        net.eval()
        with torch.no_grad():
            # iterate through the dataset
            for i in range(len(self.X)):

                # set as torch tensors
                x = torch.tensor(self.X[i, :], device=device)
                y = torch.tensor(self.y[i], device=device)

                # forward pass
                try:
                    output = net(x)
                except ValueError:
                    x = x.view(1, x.shape[0], x.shape[1], x.shape[2])
                    output = net(x)
                    
                probabilities = F.softmax(output, dim=1)

                # one hot encode the labels
                y = torch.nn.functional.one_hot(
                    y.to(torch.int64),
                    num_classes=probabilities.shape[-1],
                )

                # Now we extract the gold label and predicted true label probas

                # If the labels are binary [0,1]
                if len(torch.squeeze(y)) == 1:
                    # get true labels
                    true_probabilities = torch.tensor(probabilities)

                    # get gold labels
                    probabilities, y = torch.squeeze(
                        torch.tensor(probabilities),
                    ), torch.squeeze(y)

                    batch_gold_label_probabilities = torch.where(
                        y == 0,
                        1 - probabilities,
                        probabilities,
                    )
                # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
                elif len(torch.squeeze(y)) == 2:
                    # get true labels
                    batch_true_probabilities = torch.max(probabilities)

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities,
                        y.bool(),
                    )
                # Multi-Class labels
                else:

                    # get true labels
                    batch_true_probabilities = torch.max(probabilities)

                    # get gold labels
                    batch_gold_label_probabilities = torch.masked_select(
                        probabilities,
                        y.bool(),
                    )

                # move torch tensors to cpu as np.arrays()
                batch_gold_label_probabilities = (batch_gold_label_probabilities.cpu().numpy())
                
                batch_true_probabilities = batch_true_probabilities.cpu().numpy()

                # Append the new probabilities for the new batch
                gold_label_probabilities = np.append(gold_label_probabilities, [batch_gold_label_probabilities])
                true_probabilities = np.append(true_probabilities, [batch_true_probabilities])

        # Append the new gold label probabilities
        if self._gold_labels_probabilities is None:  # On first epoch of training
            self._gold_labels_probabilities = np.expand_dims(gold_label_probabilities,axis=-1)
        else:
            stack = [self._gold_labels_probabilities, np.expand_dims(gold_label_probabilities, axis=-1)]
            self._gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._true_probabilities is None:  # On first epoch of training
            self._true_probabilities = np.expand_dims(true_probabilities, axis=-1)
        else:
            stack = [self._true_probabilities, np.expand_dims(true_probabilities, axis=-1)]
            self._true_probabilities = np.hstack(stack)

    def on_epoch_end_batch(self, dataset, net, device="cuda:0", **kwargs):
        if isinstance(dataset, dict):
            dataset = dataset['dataloader']
        
        # compute the gradient norm
        self.gradient(net, device)

        # Compute both the gold label and true label probabilities over all samples in the dataset
        gold_label_probabilities = (list())  # gold label probabilities, i.e. actual ground truth label
        true_probabilities = list()  # true label probabilities, i.e. predicted label

        # Compute both the gold label and true label probabilities over all samples in the dataset
        adv_gold_label_probabilities = (list())  # gold label probabilities, i.e. actual ground truth label
        adv_true_probabilities = list()  # true label probabilities, i.e. predicted label

        net = net.to(device)
        net.eval()
        torch.cuda.empty_cache()
        
        # with torch.no_grad():
        # iterate through the dataset
        i=0
        
        size = len(dataset)
        
        for batch_idx, (data, target) in enumerate(dataset):
            i+=1
            # set as torch tensors
            loss_fn = torch.nn.CrossEntropyLoss()
            
            x = data.to(device)
            y = target.to(device)
            net = net.to(device)
                
            x_adv = self.get_adversarial_sample(kwargs['attack'], x, y, net, loss_fn)
            
            if isinstance(x_adv, np.ndarray):
                x_adv = torch.from_numpy(x_adv)
                x_adv = x_adv.to(device)
            net = net.to(device)
            
            # forward pass
            output = net(x)
            probabilities = F.softmax(output, dim=1)
            output = net(x_adv)
            probabilities_adv = F.softmax(output, dim=1)

            evaluation = self.evaluation.class_report(probabilities, target,True)
            self._accuracy += evaluation['accuracy']
            self._precision += evaluation['macro avg']['precision']
            self._recall += evaluation['macro avg']['recall']
            
            evaluation = self.evaluation.class_report(probabilities_adv, target,True)
            self._adv_accuracy += evaluation['accuracy']
            self._adv_precision += evaluation['macro avg']['precision']
            self._adv_recall += evaluation['macro avg']['recall']

            # one hot encode the labels
            y_adv = copy.deepcopy(y)
            y = torch.nn.functional.one_hot(y.to(torch.int64), num_classes=probabilities.shape[-1])
            y_adv = torch.nn.functional.one_hot(y_adv.to(torch.int64), num_classes=probabilities_adv.shape[-1])

            batch_gold_label_probabilities, batch_true_probabilities = self.calculate_the_probabilities(y, probabilities)
            adv_batch_gold_label_probabilities, adv_batch_true_probabilities = self.calculate_the_probabilities(y_adv, probabilities_adv)
            
            # === Normal Samples ===
            # move torch tensors to cpu as np.arrays()
            batch_gold_label_probabilities = (batch_gold_label_probabilities.detach().cpu().numpy())
            batch_true_probabilities = batch_true_probabilities.detach().cpu().numpy()

            # Append the new probabilities for the new batch
            gold_label_probabilities = np.append(gold_label_probabilities, [batch_gold_label_probabilities])
            true_probabilities = np.append(true_probabilities, [batch_true_probabilities])
            
            # === Adversarial Samples ===
            # move torch tensors to cpu as np.arrays()
            adv_batch_gold_label_probabilities = (adv_batch_gold_label_probabilities.detach().cpu().numpy())
            adv_batch_true_probabilities = adv_batch_true_probabilities.detach().cpu().numpy()

            # Append the new probabilities for the new batch
            adv_gold_label_probabilities = np.append(adv_gold_label_probabilities, [adv_batch_gold_label_probabilities])
            adv_true_probabilities = np.append(adv_true_probabilities, [adv_batch_true_probabilities])
            
            del x, y, target, data
            torch.cuda.empty_cache()
            
            print(f"\rClient: {self.client} | Batch : {batch_idx+1:03}/{size:03}", end="\r")
                
        # === Normal Samples ===
        # Append the new gold label probabilities
        if self._gold_labels_probabilities is None:  # On first epoch of training
            self._gold_labels_probabilities = np.expand_dims(
                gold_label_probabilities,
                axis=-1,
            )
        else:
            stack = [
                self._gold_labels_probabilities,
                np.expand_dims(gold_label_probabilities, axis=-1),
            ]
            self._gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._true_probabilities is None:  # On first epoch of training
            self._true_probabilities = np.expand_dims(true_probabilities, axis=-1)
        else:
            stack = [
                self._true_probabilities,
                np.expand_dims(true_probabilities, axis=-1),
            ]
            self._true_probabilities = np.hstack(stack)

        # === Adversarial Samples ===
        # Append the new gold label probabilities
        if self._adv_gold_labels_probabilities is None:  # On first epoch of training
            self._adv_gold_labels_probabilities = np.expand_dims(
                adv_gold_label_probabilities,
                axis=-1,
            )
        else:
            stack = [
                self._adv_gold_labels_probabilities,
                np.expand_dims(adv_gold_label_probabilities, axis=-1),
            ]
            self._adv_gold_labels_probabilities = np.hstack(stack)

        # Append the new true label probabilities
        if self._adv_true_probabilities is None:  # On first epoch of training
            self._adv_true_probabilities = np.expand_dims(adv_true_probabilities, axis=-1)
        else:
            stack = [
                self._adv_true_probabilities,
                np.expand_dims(adv_true_probabilities, axis=-1),
            ]
            self._adv_true_probabilities = np.hstack(stack)
            
        self._accuracy /= i
        self._precision /= i
        self._recall /= i
        self._adv_accuracy /= i
        self._adv_precision /= i
        self._adv_recall /= i 

    def calculate_the_probabilities(self, y, probabilities):
        # Now we extract the gold label and predicted true label probas

        # If the labels are binary [0,1]
        if len(torch.squeeze(y)) == 1:
            # get true labels
            batch_true_probabilities = torch.tensor(probabilities)

            # get gold labels
            probabilities, y = torch.squeeze(
                torch.tensor(probabilities),
            ), torch.squeeze(y)

            batch_gold_label_probabilities = torch.where(
                y == 0,
                1 - probabilities,
                probabilities,
            )

        # if labels are one hot encoded, e.g. [[1,0,0], [0,1,0]]
        elif len(torch.squeeze(y)) == 2:
            # get true labels
            batch_true_probabilities = torch.max(probabilities)

            # get gold labels
            batch_gold_label_probabilities = torch.masked_select(
                probabilities,
                y.bool(),
            )
        else:
            # get true labels
            batch_true_probabilities = torch.max(probabilities)

            # get gold labels
            batch_gold_label_probabilities = torch.masked_select(
                probabilities,
                y.bool(),
            )
            
        return batch_gold_label_probabilities, batch_true_probabilities
    
    def get_adversarial_sample(self, attack, x, y, net, loss_fn):
        # net.eval()
        # x.requires_grad = True
        # output = net(x)
        # loss = loss_fn(output, y)
        # loss.backward()
        
        if attack == "PGD":
            data = pgd(net, loss_fn, x, y)
            
        elif attack == "JSMA":
            data = jsma(net, loss_fn, x, y)
        
        elif attack == "FGSM":
            data = create_fgsm(x, y, net, loss_fn, alpha=0.3)
            
        return data
    
    def get_grads(self):
        """
        Returns:
            Grad norm through training: np.array(n_samples, n_epochs)
        """
        return self._grads

    def gold_labels_probabilities(self, adv=False) -> np.ndarray:
        """
        Returns:
            Gold label predicted probabilities of the "correct" label: np.array(n_samples, n_epochs)
        """
        return self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities

    def true_probabilities(self, adv=False) -> np.ndarray:
        """
        Returns:
            Actual predicted probabilities of the predicted label: np.array(n_samples, n_epochs)
        """
        return self._true_probabilities

    def confidence(self, adv=False) -> np.ndarray:
        """
        Returns:
            Average predictive confidence across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities, axis=-1)

    def aleatoric(self, adv=False):
        """
        Returns:
            Aleatric uncertainty of true label probability across epochs: np.array(n_samples): np.array(n_samples)
        """
        preds = self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities
        return np.mean(preds * (1 - preds), axis=-1)

    def variability(self, adv=False) -> np.ndarray:
        """
        Returns:
            Epistemic variability of true label probability across epochs: np.array(n_samples)
        """
        return np.std(self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities, axis=-1)

    def correctness(self, adv=False) -> np.ndarray:
        """
        Returns:
            Proportion of times a sample is predicted correctly across epochs: np.array(n_samples)
        """
        return np.mean(self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities > 0.5, axis=-1)

    def entropy(self, adv=False):
        """
        Returns:
            Predictive entropy of true label probability across epochs: np.array(n_samples)
        """
        X = self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities
        return -1 * np.sum(X * np.log(X + 1e-12), axis=-1)

    def mi(self, adv = False):
        """
        Returns:
            Mutual information of true label probability across epochs: np.array(n_samples)
        """
        T = self._gold_labels_probabilities.shape[0]
        X = self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities
        entropy = -1 * (1/T) * np.sum(X * np.log(X + 1e-12))

        X = np.mean(self._gold_labels_probabilities if not adv else self._adv_gold_labels_probabilities, axis=1)
        entropy_exp = -1 * (1/T) * np.sum(X * np.log(X + 1e-12), axis=-1)
        
        return entropy - (1/T)*entropy_exp
    
    def accuracy(self):
        return [self._accuracy, self._adv_accuracy]
    
    def recall(self):
        return [self._recall, self._adv_recall]
    
    def precision(self):
        return [self._precision, self._adv_precision]
    
    def log_results(self, client_id):
        logger.info(
            f"""Client: {client_id} \n
            Accuracy:   {self._accuracy:.4f} | {self._adv_accuracy:.4f}\n
            Recall:     {self._recall:.4f} | {self._adv_recall:.4f}\n
            Precision:  {self._precision:.4f} | {self._adv_precision:.4f}\n
            """
            # Aleatoric:  {self.aleatoric:.4f} | {self.aleatoric(adv=True):.4f}\n 
            # Predicting: {self.mi:.4f} | {self.mi(adv=True):.4f}\n 
        )