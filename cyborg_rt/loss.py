"""
loss.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F  # noqa
from torch import nn
from torch import Tensor
from torch.distributions.uniform import Uniform

from cyborg_rt.utils import prod
from cyborg_rt.utils import get_logger

logger = get_logger(__name__)


def is_image_data(input_: Tensor):
    return bool((input_.ndim == 4) and
                (input_.shape[1] == 3 or input_.shape[1] == 1))


def get_baseline(input_: Tensor, size):
    return torch.rand(
        size, *input_.shape[1:], dtype=input_.dtype).type_as(input_)

class CYBORGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, cams: Tensor, annotations: Tensor) -> Tensor:
        # be careful not to modify CAMs in-place
        cams = cams - cams.min()
        cams = cams / cams.max()
        with torch.no_grad():
            # TODO would be nice to move annotation processing logic to data
            #  loading transforms
            if is_image_data(annotations):
                # CAMs do not have a color channel. Take mean of color channels
                # to yield intensity (grayscale)
                annotations = annotations.mean(dim=1)
            annotations -= annotations.min()
            annotations /= annotations.max()
            if annotations.shape != cams.shape:
                annotations = F.interpolate(
                    annotations.unsqueeze(1), cams.shape[1:]).squeeze(1)
            annotations = annotations.detach()
        cyborg = self.mse(cams, annotations)
        return cyborg

def _downsample(image, kernel):
    return F.conv2d(image, kernel, padding='same')

def _binomial_kernel(num_channels):
    kernel = np.array((1., 4., 6., 4., 1.), dtype=np.float32)
    kernel = np.outer(kernel, kernel)
    kernel /= np.sum(kernel)
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    kernel = torch.from_numpy(kernel)
    # eye is identity matrix
    kernel =  kernel * torch.eye(num_channels, dtype=torch.float32)
    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
    # input – input tensor of shape (minibatch,in_channels,iH,iW)
    # weight – filters of shape (out_channels,in_channels/groups,kH,kW
    kernel = kernel.swapaxes(0,2)
    kernel = kernel.swapaxes(1,3)
    return kernel

def pyramidal_representation(heatmap_image, num_levels):
    kernel = _binomial_kernel(heatmap_image.shape[3])
    levels = [heatmap_image]
    # input image needs same format as kernel
    heatmap_image = heatmap_image.swapaxes(1,3)
    kernel = kernel.type_as(heatmap_image)
    for i in range(num_levels):
        heatmap_image = _downsample(heatmap_image, kernel)
        levels.append(heatmap_image)
    return levels

# deal with tokens later
def pyramidal_mse_with_tokens(true_heatmaps, predicted_heatmaps, tokens=None, nb_levels=5):
    pyramid_y      = pyramidal_representation(true_heatmaps[:, :, :, None], nb_levels)
    pyramid_y_pred = pyramidal_representation(predicted_heatmaps[:, :, :, None], nb_levels)

    loss = torch.mean(torch.stack([
                # torch mse instead for all levels
                # instead of their tokenized MSE
                # in future, we can tokenize the MSE 
                # with the reaction times potentially
                F.mse_loss(pyramid_y[i], pyramid_y_pred[i])
                for i in range(nb_levels)]))
    return loss


# TODO: we modified the wrong one 
# correct this 
# and also the harmonization loss original also uses the 
# same cross-entropy calculation
class HarmonizationCYBORGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        # self.mse = nn/.MSELoss()

    def forward(self, output: Tensor, target: Tensor,
            cams: Tensor, annotations: Tensor) -> Tensor: 
        # be careful not to modify CAMs in-place
        cams = cams - cams.min()
        cams = cams / cams.max()

        # NOTE
        # might need to mess around with the gradients here at various
        # points, given the nature of tf.GradientTape vs Lit

        with torch.no_grad():
            if is_image_data(annotations):
                # CAMs do not have a color channel. Take mean of color channels
                # to yield intensity (grayscale)
                annotations = annotations.mean(dim=1)
            annotations -= annotations.min()
            annotations /= annotations.max()

            if annotations.shape != cams.shape:
                annotations = F.interpolate(
                    annotations.unsqueeze(1), cams.shape[1:]).squeeze(1)
            annotations = annotations.detach()

            # standardized cut procedure
            # from the harmonization paper, we only 
            # take the positive values, via a relu function
            # this is in addition to the previous normalization steps
            # which CYBORG also uses
            # to positive values 
            annotations = F.relu(annotations)
            cams = F.relu(cams)

            # TODO: questionable intermediate normalization ...
            # might need to change to torch-like representation
            # re-normalize before pyramidal
            # _annotations_max = torch.max(annotations, (1, 2), keepdims=True) + 1e-6
            _annotations_max = annotations.max()
            # _cam_max = torch.detach(torch.max(cams, (1, 2), keepdims=True))  + 1e-6
            _cam_max = cams.max()
            # normalize the true heatmaps according to the saliency maps
            annotations = annotations / _annotations_max * _cam_max

            # model preds
            _, pred = torch.max(output.data, 1)

        # Gaussian pyramid
        harmonization_loss = pyramidal_mse_with_tokens(annotations,
                                                       cams)
        
        ce_output = self.ce(output, target)

        # TODO: weight loss? not sure what this does
        
        # NOTE: might be worth trying the alpha param here, too
        return harmonization_loss * ce_output


class CYBORGBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        assert 0 < alpha < 1
        self.alpha = alpha

        self.bce = nn.BCEWithLogitsLoss()
        self.cyborg = CYBORGLoss()

    def forward(self, output: Tensor, target: Tensor, cams: Tensor,
                annotations: Tensor) -> Tensor:
        bce = self.bce(output, target)
        cyborg = self.cyborg(cams=cams, annotations=annotations)
        return self.alpha * bce + (1.0 - self.alpha) * cyborg


class CYBORGCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        assert 0 < alpha < 1
        self.alpha = alpha

        self.ce = nn.CrossEntropyLoss()
        self.cyborg = CYBORGLoss()

    def forward(self, output: Tensor, target: Tensor, cams: Tensor,
                annotations: Tensor) -> Tensor:
        ce = self.ce(output, target)
        cyborg = self.cyborg(cams=cams, annotations=annotations)
        return self.alpha * ce + (1.0 - self.alpha) * cyborg

class CYBORGCrossEntropyLossXReactionTime(nn.Module):
    """
    Justin Dulay - 12/07/2022

    Note - for using placeholder:
    Creating a placeholder for adding reaction time to CYBORG Loss
    self.rt is a torch tensor of normally distributed reaction times 
    from [1,20]. These are reflective of presumptive annotation
    times, i.e. they are my qualitative guess. 
    
    Config.py
    LOSS = CYBORG+REACTIONTIME

    Update -  01/03/2023

    Need to change loss such that:
    - penalties are added to cross-entropy tensor
    - TODO: scaling occurs more correctly
    
    """
    def __init__(self, alpha=0.5, psych_scaling_constant=0.1):
        super().__init__()
        assert 0 < alpha < 1
        self.alpha = alpha
        self.psych_scaling_constant = psych_scaling_constant

        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.cyborg = CYBORGLoss()

    def forward(self, output: Tensor, target: Tensor, reaction_times: Tensor,
                cams: Tensor, annotations: Tensor) -> Tensor:
        cyborg = self.cyborg(cams=cams, annotations=annotations)
        # create a psych penalty on a sample index level
        # if there is no penalty, the sample is just multiplied by 1
        psych_penalties = torch.ones(len(output)).type_as(reaction_times)

        lower = torch.quantile(reaction_times, 0.25).item()
        upper = torch.quantile(reaction_times, 0.75).item()
        # change values inside lower and upper quartiles to 0/1
        for i in range(len(reaction_times)):
            if reaction_times[i].item() < lower:
                reaction_times[i] = 0
            elif reaction_times[i].item() > upper:
                reaction_times[i] = 1
        
        # this is inefficient, but precise for now
        # find min and max elements in tensor that are not 0 or 1 
        min_elem = 100.0
        max_elem = -100.0
        for i in range(len(reaction_times)):
            if reaction_times[i] < min_elem and (reaction_times[i] != 0.0 and reaction_times[i] != 1.0):
                min_elem = reaction_times[i].item()
            if reaction_times[i] > max_elem and (reaction_times[i] != 1.0 and reaction_times[i] != 0.0):
                max_elem = reaction_times[i].item()

        # now we normalize just the values in reaction times piece wise
        # batchwise normalize to [0, 1] along with height and width
        for i in range(len(reaction_times)):     
            if reaction_times[i].item() != 0 and reaction_times[i].item() != 1: 
                reaction_times[i] -= min_elem
                reaction_times[i] /= max_elem

        # inference on model predictions
        _, preds = torch.max(output.data, 1)
        for i in range(len(preds)):
            if preds[i] != target[i]:
                psych_penalties[i] *= reaction_times[i]
                # output[i] += psych_penalties[i].clone() 
        
        # regular cross-entropy loss after we have modified the logits with reactiontimes
        ce = self.ce(output, target)

        # add psych penalties to loss forward pass
        for i in range(len(preds)):
            if preds[i] != target[i]:
                ce[i] += psych_penalties[i].clone()
        # then reduce to common value (mean is default)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # i.e. the reduce_mean of the minibatch
        ce = ce.mean()

        # return cyborg weighted loss
        return self.alpha * ce + (1.0 - self.alpha) * cyborg

class ReactionTime(nn.Module):
    """
    Justin Dulay - 12/20/2022

    Config.py
    LOSS = REACTIONTIME

    Update -  01/03/2023

    Need to change loss such that:
    - penalties are added to cross-entropy tensor
    - TODO: scaling occurs more correctly
    """
    def __init__(self, psych_scaling_constant=0.1):
        super().__init__()
        self.psych_scaling_constant = psych_scaling_constant
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output: Tensor, target: Tensor, reaction_times: Tensor) -> Tensor:
        # create a psych penalty on a sample index level
        # if there is no penalty, the sample is just multiplied by 1
        psych_penalties = torch.ones(len(output)).type_as(reaction_times)

        # calculate quartiles for reaction times
        lower = torch.quantile(reaction_times, 0.25).item()
        upper = torch.quantile(reaction_times, 0.75).item()
        # change values inside lower and upper quartiles to 0/1
        for i in range(len(reaction_times)):
            if reaction_times[i].item() < lower:
                reaction_times[i] = 0
            elif reaction_times[i].item() > upper:
                reaction_times[i] = 1
        
        # this is inefficient, but precise for now
        # find min and max elements in tensor that are not 0 or 1 
        min_elem = 100.0
        max_elem = -100.0
        for i in range(len(reaction_times)):
            if reaction_times[i] < min_elem and (reaction_times[i] != 0.0 and reaction_times[i] != 1.0):
                min_elem = reaction_times[i].item()
            if reaction_times[i] > max_elem and (reaction_times[i] != 1.0 and reaction_times[i] != 0.0):
                max_elem = reaction_times[i].item()
        # now we normalize just the values in reaction times piece wise
        # batchwise normalize to [0, 1] along with height and width
        for i in range(len(reaction_times)):     
            if reaction_times[i].item() != 0 and reaction_times[i].item() != 1: 
                reaction_times[i] -= min_elem
                reaction_times[i] /= max_elem

        # inference on model predictions to calculate 
        # which indices to applly psych_penalties
        _, preds = torch.max(output.data, 1)
        for i in range(len(preds)):
            if preds[i] != target[i]:
                # psych_penalties[i] *= reaction_times[i] * self.psych_scaling_constant
                # no need for scaling constant anymore
                psych_penalties[i] *= reaction_times[i]
                # output[i] += psych_penalties[i].clone() 

        # regular cross-entropy
        ce = self.ce(output, target)

        # add psych penalties to loss forward pass
        for i in range(len(preds)):
            if preds[i] != target[i]:
                ce[i] += psych_penalties[i].clone()
        # then reduce to common value (mean is default)
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        # i.e. the reduce_mean of the minibatch
        ce = ce.mean()

        return ce