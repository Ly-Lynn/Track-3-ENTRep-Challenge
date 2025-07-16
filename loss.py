from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class InfoNCELoss(Loss):
    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def __call__(self, image_embeds, text_embeds, label_smoothing=0.1):
        logits = torch.matmul(image_embeds, text_embeds.transpose(0, 1)) / self.temperature
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        image_loss = nn.functional.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        text_loss = nn.functional.cross_entropy(logits.transpose(0, 1), labels, label_smoothing=label_smoothing)

        return (image_loss + text_loss) / 2