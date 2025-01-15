import numpy, scipy,logging
import numpy as np
import torch
from sklearn.metrics import *

logger = logging.getLogger(__name__)
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0.5, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model, key, snr):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, key, snr)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, val_loss, model, key, snr ):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        try:
            torch.save(model, './models/{}_expertSNR{}.pth'.format(key, snr))
            torch.save(model.state_dict(), './models/{}_expert_weightsSNR{}.pth'.format(key, snr))
        except:
            logger.error('Something went wrong while saving {} Expert'.format(key))
        self.val_loss_min = val_loss          

class Evaluation():
    def __init__(self):
        self.accuracy = []
        self.precision = []
        self.matrix = 0
        self.recall = []
        self.classes = []
        
    def gather_info(self, predicted, actual):
        try:
            self.matrix += confusion_matrix(actual.cpu().detach().numpy(), torch.max(predicted.cpu().detach(), 1)[1].numpy())
        except:
            pass
        
        report = classification_report(actual.cpu().detach().numpy(), torch.max(predicted.cpu().detach(), 1)[1].numpy(), output_dict=True)
        self.accuracy.append(report['accuracy'])
        self.precision.append(report['macro avg']['precision'])
        self.recall.append(report['macro avg']['recall'])
        
    def log_the_info(self):
        logger.info('Accuracy Score: {:.3f}% \nPrecision Score: {:.3f}% \nRecall Score: {:.3f}% \n'.format(
            100*sum(self.accuracy)/len(self.accuracy), 
            100*sum(self.precision)/len(self.precision),
            100*sum(self.recall)/len(self.recall)))

class EvaluationConvergence():
    def __init__(self, client_id):
        self.matrix = 0
        self.report = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.log_results = True
        self.client_id = client_id
        
    def gather_info(self, predicted, actual):
        self.accuracy = 100*accuracy_score(actual.cpu().detach(), torch.max(predicted.cpu().detach(), 1)[1].numpy())
        self.recall = 100*recall_score(actual.cpu().detach(), torch.max(predicted.cpu().detach(), 1)[1].numpy(), average="macro")
        self.precision = 100*precision_score(actual.cpu().detach(), torch.max(predicted.cpu().detach(), 1)[1].numpy(), average="macro",)
        self.report = classification_report(actual.cpu().detach().numpy(), torch.max(predicted.cpu().detach(), 1)[1].numpy())
        
        try:
            self.matrix += confusion_matrix(actual.cpu().detach().numpy(), torch.max(predicted.cpu().detach(), 1)[1].numpy())
        except:
            pass
        
        if self.log_results:
            self.log_info()
    
    def log_info(self):
        logger.info('Client {}:\nClassification Report: \n{}'.format(self.client_id, self.report))
        logger.info('Matrix: \n{}'.format(self.matrix))
        # logger.info(' Accuracy Score: {:.4f}% \n Precision Score: {:.4f}% \n Recall Score: {:.4f}% \n'.format(self.accuracy, self.precision,self.recall))

class EvaluationMetrices():
    def __init__(self):
        self.matrix = 0
        self.report = ''
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.length = 0

    def calculate(self, predicted, actual):
        self.conf_matrix(predicted, actual)
        self.class_report(predicted, actual)
        self.estimate_accuracy(predicted, actual)
        self.estimate_precision(predicted, actual)
        self.estimate_recall(predicted, actual)
        
    def log_evaluation(self, batch):
        self.length=batch
        self.accuracy /=self.length
        self.precision /=self.length
        self.recall /=self.length
        
        self.log_class_report()
        self.log_conf_matrix()
        self.log_apr()
        # self.log_accuracy()
        # self.log_precision()
        # self.log_recall()

    def conf_matrix(self, predicted, actual): 
        try:
            self.matrix += confusion_matrix(actual.cpu().detach().numpy(), torch.max(predicted.cpu().detach(), 1)[1].numpy())
        except:
            pass
        
    def log_conf_matrix(self):
        logger.info('Matrix: \n{}'.format(self.matrix))
        
    def class_report(self, predicted, actual, output_dict = False):    
        self.report = classification_report(actual.cpu().detach().numpy(), torch.max(predicted.cpu().detach(), 1)[1].numpy(), output_dict=output_dict)
        
        if output_dict:
            return self.report
        
    def log_class_report(self):
        logger.info('Classification Report: \n{}'.format(self.report))
        
    def estimate_accuracy(self, predicted, actual):
        self.accuracy += 100*accuracy_score(actual.cpu().detach(), torch.max(predicted.cpu().detach(), 1)[1].numpy())
    
    def log_accuracy(self):
        logger.info('Accuracy Score: \n{:.4f}%'.format(self.accuracy))
        
    def estimate_precision(self, predicted, actual):
        self.precision += 100*precision_score(actual.cpu().detach(), torch.max(predicted.cpu().detach(), 1)[1].numpy(), average="macro",)
        
    def log_precision(self):
        logger.info('Precision Score: \n{:.4f}%'.format(self.precision))

    def estimate_recall(self, predicted, actual):
        self.recall += 100*recall_score(actual.cpu().detach(), torch.max(predicted.cpu().detach(), 1)[1].numpy(), average="macro")
        
    def log_recall(self):
        logger.info('Recall Score: \n{:.4f}%'.format(self.recall))
        
    def log_accuracy_precion_recall(self):
        logger.info(' Accuracy Score: {:.4f}% \n Precision Score: {:.4f}% \n Recall Score: {:.4f}% \n'.format(self.accuracy, self.precision,self.recall))
