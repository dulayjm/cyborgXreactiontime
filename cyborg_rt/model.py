"""
models.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""
from functools import partial

import torch
from torch import Tensor
from torch import nn
import torchvision.models as models
from torchmetrics import Accuracy
from torchmetrics import AUROC
from torchmetrics import AveragePrecision
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall
from pytorch_lightning import LightningModule
import wandb

from cyborg_rt.loss import CYBORGBCEWithLogitsLoss
from cyborg_rt.loss import CYBORGCrossEntropyLoss
from cyborg_rt.loss import CYBORGCrossEntropyLossXReactionTime
from cyborg_rt.loss import ReactionTime
from cyborg_rt.loss import HarmonizationCYBORGLoss
from cyborg_rt.utils import get_logger
from cyborg_rt.utils import requires_human_annotations

logger = get_logger(__name__)


class DualGateResNet50(nn.Module):
    def __init__(self, pretrained, num_classes) -> None:
        super().__init__()
        backbone = models.resnet50(pretrained=pretrained)
        print('debug default resnet config', backbone)
        self.in_features = backbone.fc.in_features
        # pop off the last layer of the model
        self.model = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        self.classifier = nn.Linear(self.in_features, num_classes)
        self.regressor = nn.Linear(self.in_features, 1)
        
    def forward(self, x) -> Tensor:
        print('is this empty?', self.model)
        print('starting forward pass')
        print('in features are', self.in_features)
        print('self.classifier is ', self.classifier)
        logits = self.model(x)
        print('logits are', logits.shape)

        logits = logits.view(-1)
        print('logits are', logits.shape)
        # why are we not going through this step
        class_outputs = self.classifier(logits)
        print('class_outputs are', class_outputs.shape)
        regression_outputs = self.regressor(logits)
        print('regression_outputs are', regression_outputs.shape)
        return class_outputs, regression_outputs

def disable_grad(model):
    for param in model.parameters():
        param.requires_grad = False

def get_backbone(name: str, num_classes: int, pretrained=True, freeze=False):
    """
    Networks are instantiated from the pre-trained ImageNet weights
    """
    if name in {'Xception', 'CNNDetection', 'Self-Attention'}:
        # TODO
        #  CNNDetection: https://github.com/peterwang512/CNNDetection
        #  Xception: https://github.com/Cadene/pretrained-models.pytorch
        #  Self-Attention: https://github.com/JStehouwer/FFD_CVPR2020
        raise NotImplementedError(name)
    elif name == 'ResNet50':
        pretrained_model = models.resnet50(pretrained=pretrained)
        if freeze:
            disable_grad(pretrained_model)
        in_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Linear(in_features, num_classes)
    elif name == 'DualGateResNet50':
        pretrained_model = DualGateResNet50(pretrained, num_classes)
        if freeze:
            disable_grad(pretrained_model)
        # feature engineering taken care of in the model constructor
    elif name == 'DenseNet121':
        pretrained_model = models.densenet121(pretrained=pretrained)
        if freeze:
            disable_grad(pretrained_model)
        in_features = pretrained_model.classifier.in_features
        pretrained_model.classifier = nn.Linear(in_features, num_classes)
    elif name == 'Inception_v3':
        pretrained_model = models.inception_v3(pretrained=pretrained)
        if freeze:
            disable_grad(pretrained_model)
        # Handle the auxiliary net
        if pretrained_model.AuxLogits is not None:
            in_features = pretrained_model.AuxLogits.fc.in_features
            pretrained_model.AuxLogits.fc = nn.Linear(in_features, num_classes)
        # Handle the primary net
        in_features = pretrained_model.fc.in_features
        pretrained_model.fc = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f'Unsupported model "{name}"')

    return pretrained_model


def get_last_conv(name: str, backbone: nn.Module) -> nn.Module:
    if name in {'Xception', 'CNNDetection', 'Self-Attention'}:
        # TODO
        raise NotImplementedError(name)
    elif name == 'ResNet50' or name == 'DualGateResNet50':
        return backbone.layer4
    elif name == 'DenseNet121':
        return backbone.features
    elif name == 'Inception_v3':
        return backbone.Mixed_7c
    else:
        raise ValueError(f'Unsupported model "{name}"')


def get_readout(name: str, backbone: nn.Module) -> nn.Linear:
    if name in {'Xception', 'CNNDetection', 'Self-Attention'}:
        # TODO
        raise NotImplementedError(name)
    elif name == 'ResNet50':
        return backbone.fc
    elif name == 'DualGateResNet50':
        return backbone.classifier
    elif name == 'DenseNet121':
        return backbone.classifier
    elif name == 'Inception_v3':
        return backbone.fc
    else:
        raise ValueError(f'Unsupported model "{name}"')


def get_input_size(name: str):
    return {
        'ResNet50': 224,
        'DualGateResNet50': 224,
        'DenseNet121': 224,
        'Inception_v3': 299,
    }[name]


def get_optimizer(name: str):
    return getattr(torch.optim, name)


# noinspection PyAbstractClass
class CYBORGxSAL(LightningModule):
    def __init__(self, C):
        super().__init__()

        self.C = C
        print('on init, the cnofig is', self.C)
        # binary classification task
        num_classes = 1 if C.BINARY_OUTPUT else 2

        # init a pretrained model
        self.backbone = get_backbone(C.BACKBONE, num_classes=num_classes)
        logger.info(f'Backbone: {C.BACKBONE}')

        # loss
        loss = C.LOSS.upper()
        self.criterion_requires_model = False
        self.criterion_requires_input = False
        self.criterion_requires_cams = False
        self.criterion_requires_reactiontimes = False
        self.criterion_uses_harmonization = False

        if loss in {'BCE', 'CE'}:
            self.criterion = (nn.BCEWithLogitsLoss() if C.BINARY_OUTPUT else
                              nn.CrossEntropyLoss())
        elif loss == 'CYBORG':
            loss_cls = (CYBORGBCEWithLogitsLoss if C.BINARY_OUTPUT else
                        CYBORGCrossEntropyLoss)
            self.criterion = loss_cls(
                alpha=C.CYBORG_LOSS_ALPHA,
            )
            self.criterion_requires_cams = True
        elif loss == 'CYBORG+REACTIONTIME':
            loss_cls = CYBORGCrossEntropyLossXReactionTime
            self.criterion = loss_cls(
                alpha=C.CYBORG_LOSS_ALPHA,
                psych_scaling_constant=C.PSYCH_SCALING_CONSTANT
            )
            self.criterion_requires_cams = True 
            self.criterion_requires_reactiontimes = True
        elif loss == 'CYBORG+HARMONIZATION':
            self.criterion = HarmonizationCYBORGLoss()
            self.criterion_requires_cams = True 
            self.criterion_requires_reactiontimes = False
            self.criterion_uses_harmonization = True
        elif loss == 'REACTIONTIME':
            loss_cls = ReactionTime
            self.criterion = loss_cls(
                psych_scaling_constant=C.PSYCH_SCALING_CONSTANT
            )
            self.criterion_requires_reactiontimes = True
        elif loss == 'DIFFERENTIABLE_REACTIONTIME':
            # for now, the classifier loss is the criterion here
            # the loss component of RT is setup just in the loss itself 
            # and the criterion_requires_reactiontimes enables the kwargs
            self.criterion = nn.CrossEntropyLoss()
            self.criterion_requires_reactiontimes = True
        elif loss in {'CYBORG+SAL', 'SAL+CYBORG', 'SAL'}:
            self.criterion = None
            raise NotImplementedError(loss)
        else:
            raise ValueError(f'Unknown loss {C.LOSS}')

        # metrics
        splits = ['train', 'val', 'test']
        metric_defs = {
            'auc': (AUROC, dict(task='multiclass',
                                num_classes=num_classes,
                                average='macro',
                                compute_on_step=False)),
            'ap': (AveragePrecision, dict(task='multiclass',
                                          num_classes=num_classes,
                                          average='macro',
                                          compute_on_step=False)),
            'accuracy': (Accuracy, dict(task='multiclass',
                                        num_classes=num_classes,
                                        threshold=0.5,
                                        average='micro',
                                        compute_on_step=False)),
            'f1': (F1Score, dict(task='multiclass',
                            num_classes=num_classes,
                            threshold=0.5,
                            average='micro',
                            compute_on_step=False)),
            'precision': (Precision, dict(task='multiclass',
                                          num_classes=num_classes,
                                          threshold=0.5,
                                          average='micro',
                                          compute_on_step=False)),
            'recall': (Recall, dict(task='multiclass',
                                    num_classes=num_classes,
                                    threshold=0.5,
                                    average='micro',
                                    compute_on_step=False)),
        }
        self.metrics = {}
        for split in splits:
            metrics_split = {}
            for metric_name, (metric_cls, metric_kwargs) in metric_defs.items():
                metrics_split[metric_name] = metric_cls(**metric_kwargs)
                # now we set attributes so that way lightning can auto-detect
                #  the metrics
                attr_name = f'metric_{metric_name}_{split}'
                if getattr(self, attr_name, None) is not None:
                    raise RuntimeError(
                        f'The attribute {attr_name} is already set!')
                setattr(self, attr_name, metrics_split[metric_name])
            self.metrics[split] = metrics_split

        self.metrics_test = {}
        for dataset_idx in range(C.TEST_DATASET_IDXS):
            metrics_test_idx = {}
            for name, metric in self.metrics['test'].items():
                metrics_test_idx[name] = metric.clone()
                attr_name = f'metric_{name}_test_{dataset_idx}'
                if getattr(self, attr_name, None) is not None:
                    raise RuntimeError(
                        f'The attribute {attr_name} is already set!')
                setattr(self, attr_name, metrics_test_idx[name])
            self.metrics_test[dataset_idx] = metrics_test_idx

    def configure_optimizers(self):
        optimizer_cls = get_optimizer(self.C.OPTIMIZER)
        params_to_update = []
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                logger.info(f'Optimizer will update parameters of "{name}"')
                params_to_update.append(param)
        # TODO: you might need to change params_to_update for optimizer2
        optimizer = optimizer_cls(params_to_update,
                                  lr=self.C.LEARNING_RATE,
                                  weight_decay=self.C.WEIGHT_DECAY,
                                  momentum=self.C.MOMENTUM)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.C.LEARNING_RATE_DECAY_STEP_SIZE,
            gamma=self.C.LEARNING_RATE_DECAY_GAMMA,
        )
        return [optimizer], [scheduler]

    def forward(self, x, prefix='test'):
        # special case for inception training outputs
        if self.C.BACKBONE == 'Inception_v3' and prefix=='train':
            out, aux = self.backbone(x) 
        elif self.C.LOSS == 'DIFFERENTIABLE_REACTIONTIME' \
            and self.C.BACKBONE == 'DualGateResNet50' \
            and prefix=='train':
            out, regression_output = self.backbone(x)
            return out, regression_output
        else:
            out = self.backbone(x)

        if self.C.BINARY_OUTPUT:
            out = torch.squeeze(out, dim=1)
        return out

    def _compute_criterion(self, logits, target, input_, **kwargs):
        if self.criterion_requires_model:
            kwargs['input_'] = input_
        if self.criterion_requires_input:
            kwargs['model'] = self
        if self.C.BINARY_OUTPUT:
            target = target.float()

        loss = self.criterion(logits, target, **kwargs)
        return loss

    def _compute_psych_criterion(logits, target, **kwargs):
        loss = ...
        return loss

    def on_train_start(self):
        if self.C.DEBUG:
            torch.autograd.set_detect_anomaly(True)

        if not self.criterion_requires_cams:
            return

        last_conv = get_last_conv(self.C.BACKBONE, self.backbone)

        def save_last_conv(module, input_, output):
            if not self.training:
                return
            self.last_conv_output = output

        self.last_conv_output = None
        self.last_conv_hook = last_conv.register_forward_hook(save_last_conv)

        self.backbone_classifier = get_readout(self.C.BACKBONE, self.backbone)

    def training_step(self, batch, batch_idx, dataset_idx=None, optimizer_idx=None):
        # TODO: refactor these config steps
        # CYBORG
        if requires_human_annotations(self.C) \
            and not self.criterion_requires_reactiontimes \
            and not self.criterion_uses_harmonization:
            x, y, annotations = batch
            # print('we are at regular cyborg')
            kwargs = {'annotations': annotations}
        # CYBORG+REACTIONTIME
        elif requires_human_annotations(self.C) \
            and self.criterion_requires_reactiontimes:
            x, y, annotations, reaction_times = batch
            # print("We are at cyborg+reacotoin")

            kwargs = {'annotations': annotations}
            kwargs['reaction_times'] = reaction_times
        # REACTIONTIME
        elif not requires_human_annotations(self.C) \
            and self.criterion_requires_reactiontimes:
            print("We are at reactiomntime weird config thing")
            x, y, reaction_times = batch
            kwargs = {'reaction_times': reaction_times}
        # CYBORG+HARMONIZATION
        elif requires_human_annotations(self.C) \
            and not self.criterion_requires_reactiontimes \
            and self.criterion_uses_harmonization:
            # print("We are at harmonization")
            x, y, annotations = batch
            kwargs = {'annotations': annotations}
        else:
            x, y = batch
            kwargs = {}

        # forward pass
        if self.C.BACKBONE == 'DualGateResNet50' and self.C.LOSS == 'DIFFERENTIABLE_REACTIONTIME':
            logits, regression_logits = self(x, prefix='train')
        else:
            logits = self(x, prefix='train')

        if self.criterion_requires_cams:
            # print('we are requiring cams')
            # Compute CAMs
            if self.C.BINARY_OUTPUT:
                cams = torch.tensordot(
                    self.backbone_classifier.weight[0],
                    self.last_conv_output,
                    dims=[(0,), (1,)],
                )
                # negate CAMs for class 0 since there is only one class.
                # otherwise, we would instead be computing the CAM of each
                # correct class. the annotations for class 1 indicate where
                # humans thought the image indicated a deepfake. the annotations
                # for class 0 indicate where humans thought the image indicated
                # real. therefore, we want the CAMs to align in the negative
                # direction (which pushes classification towards 0)
                cams[y == 0] *= -1
            else:
                assert self.backbone_classifier.weight.shape[0] == 2
                cams_per_class = {}
                for i in range(2):
                    cams_per_class[i] = torch.tensordot(
                        self.backbone_classifier.weight[i],
                        self.last_conv_output,
                        dims=[(0,), (1,)],
                    )
                y_expanded = y.view(len(y),
                                    *([1] * (cams_per_class[0].ndim - 1)))
                cams = torch.where(y_expanded == 0,
                                   cams_per_class[0],  # class 0 cams
                                   cams_per_class[1])  # class 1 cams
            kwargs['cams'] = cams
        
        # generate reaction times 
        if self.criterion_requires_reactiontimes and self.C.USE_RANDOM_REACTIONTIME == 'random':
            # place_holder dummy reaction times
            # reaction times are random variables in toy example
            # as many as the batch size 
            kwargs['reaction_times'] = None
            dummy_reaction_times = torch.randint(1,20,(len(y),)).type_as(x)
            kwargs['reaction_times'] = dummy_reaction_times

        if self.C.BACKBONE == 'DualGateResNet50' and self.C.LOSS == 'DIFFERENTIABLE_REACTIONTIME':
            class_loss = self._compute_criterion(logits, y, x, **kwargs)
            reaction_times = kwargs['reaction_times'].type_as(regression_logits)
            psych_loss = self._compute_psych_criterion(regression_logits, reaction_times)
            #TODO: optimize this somehow
            loss = class_loss + psych_loss
        else:
            loss = self._compute_criterion(logits, y, x, **kwargs)
        
        self.log('train/loss', loss)
        return loss

    def on_train_end(self):
        if self.criterion_requires_cams:
            self.last_conv_hook.remove()
            self.last_conv_output = None
            self.backbone_classifier = None

    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        if self.criterion_requires_cams:
            # ensure these things don't end up in state dict
            bad_keys = [*filter(lambda k: k.startswith('backbone_classifier'),
                                d.keys())]
            for bad_key in bad_keys:
                del d[bad_key]
        return d

    def validation_step(self, batch, batch_idx, dataset_idx=None):
        self._shared_eval(batch, batch_idx, dataset_idx, 'val')

    def test_step(self, batch, batch_idx, dataset_idx=None):
        self._shared_eval(batch, batch_idx, dataset_idx, 'test')

    # def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
    #     dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
    #     model_filename = "model_final.onnx"
    #     self.to_onnx(model_filename, dummy_input, export_params=True)
    #     artifact = wandb.Artifact(name="model.ckpt", type="model")
    #     artifact.add_file(model_filename)
    #     wandb.log_artifact(artifact)

    def _shared_eval(self, batch, batch_idx, dataset_idx, prefix,
                     criterion=True):
        if dataset_idx is not None and prefix != 'test':
            raise NotImplementedError(f'Multiple datasets for {prefix} '
                                      'split are not supported yet.')
        x, y = batch[:2]

        logits = self(x, prefix=prefix)

        # NOTE: I don't care about BCE right now. 
        # if criterion and not requires_human_annotations(self.C):
        #     # BCE criterion has fused sigmoid/softmax
        #     loss = self._compute_criterion(logits, y, x)
        #     self.log(f'{prefix}_loss', loss)

        activation = (torch.sigmoid if self.C.BINARY_OUTPUT else
                      partial(torch.softmax, dim=1))
        scores = activation(logits)
        for name, metric in self.metrics[prefix].items():
            if dataset_idx is not None:
                # retrieve metric for this particular dataset index
                metric = self.metrics_test[dataset_idx][name]
            metric.update(scores, y)
            self.log(f'{prefix}_{name}', metric, on_step=False, on_epoch=True,
                     prog_bar=True)