import torch
from detectron2.utils import comm
from sklearn.metrics import roc_auc_score
import numpy as np


class ObjectnessMetrics:
    def __init__(self, cfg):
        """
        Sum, Mean and standard deviation of the output of final RPN linear layer
        """
        self.output_sum = torch.Tensor().to(cfg.MODEL.DEVICE)
        self.output_mean = -1
        self.output_std = -1

        """
        Sum and mean of the sigmoidal output (score) of final RPN linear layer
        """
        self.sum_score = torch.Tensor().to(cfg.MODEL.DEVICE)
        self.mean_score = -1

    def update(self, output):
        self.output_sum = torch.cat((self.output_sum, output))
        self.sum_score = torch.cat((self.sum_score, torch.sigmoid(output)))

    def calculate_metrics(self):
        list_output_sum = comm.gather(self.output_sum, dst=0)
        list_sum_score = comm.gather(self.sum_score, dst=0)
        if not comm.is_main_process():
            return
        total_output_sum = torch.Tensor().to("cpu")
        total_sum_score = torch.Tensor().to("cpu")

        for output_sum, output_score in zip(list_output_sum, list_sum_score):
            total_output_sum = torch.cat((total_output_sum, output_sum.to("cpu")))
            total_sum_score = torch.cat((total_sum_score, output_score.to("cpu")))

        self.output_mean = torch.mean(total_output_sum)
        self.output_std = torch.std(total_output_sum)
        self.mean_score = torch.mean(total_sum_score)


class ActivationMetrics:
    def __init__(self):
        """
        Mean and standard deviation of the activation of final Fast RCNN final layer
        """
        self.accumulator = torch.Tensor()
        self.mean = None
        self.std = None

    def update(self, logit):
        self.accumulator = torch.cat((self.accumulator, logit))

    def calculate_metrics(self):
        if len(self.accumulator) == 0:
            mean = torch.Tensor(0)
        elif len(self.accumulator) == 1:
            std = torch.Tensor(0)
            mean = self.accumulator.mean()
        else:
            mean = self.accumulator.mean()
            std = self.accumulator.std()
        self.mean = mean
        self.std = std


class DetectionMetrics:

    @staticmethod
    def ap(rec, prec):
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @staticmethod
    def auroc(y_true, y_scores):
        roc_auc_score(y_true=y_true, y_scores=y_scores)
