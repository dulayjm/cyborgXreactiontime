"""
data.py - A CYBORG-SAL-Collab file
Copyright (C) 2021  Zach Carmichael
"""
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torchvision.transforms import Resize
from torchvision.transforms import CenterCrop
from torchvision.transforms import Compose
from torchvision.datasets.folder import default_loader as image_loader
from pytorch_lightning import LightningDataModule
import pandas as pd
import numpy as np

from cyborg_rt.utils import get_logger
from cyborg_rt.utils import num_cpus
from cyborg_rt.utils import dict_union
from cyborg_rt.utils import requires_human_annotations
from cyborg_rt.model import get_input_size

logger = get_logger(__name__)


class QuickLabels:
    """Quick label fetching in the case that label values are grouped by label
    in a sequence"""

    __slots__ = '_values', '_lengths', '_cumsums'

    def __init__(self, values, lengths):
        if isinstance(lengths, int):
            lengths = [lengths]
            values = [values]
        assert len(values) == len(lengths), f'{len(values)} != {len(lengths)}'
        assert len(values) > 0, 'No values or lengths provided!'
        self._values = values
        self._lengths = lengths
        self._cumsums = [lengths[0]]
        prev_sum = lengths[0]
        for length in lengths[1:]:
            cumsum = prev_sum = prev_sum + length
            self._cumsums.append(cumsum)

    def __len__(self):
        return self._cumsums[-1]

    def _handle_bad_index(self, item):
        raise IndexError(f'Index {item} is out of bounds for '
                         f'{self.__class__.__name__} with length '
                         f'{len(self)}!')

    def __iter__(self):
        for value, length in zip(self._values, self._lengths):
            for _ in range(length):
                yield value

    def __getitem__(self, item):
        assert isinstance(item, int)
        if item < 0:
            item_ = item + len(self) - 1
            if item_ < 0:
                self._handle_bad_index(item)
            item = item_
        for item_idx, cumsum in enumerate(self._cumsums):
            if item < cumsum:
                break
        else:
            self._handle_bad_index(item)
        return self._values[item_idx]

    def __setitem__(self, key, value):
        raise TypeError(f'{self.__class__.__name__} does not support '
                        'setting items!')

    def __str__(self):
        return '[' + ', '.join(f'{value}×{length}'
                               for value, length in
                               zip(self._values, self._lengths)) + ']'

    def __repr__(self):
        return str(self)


def get_transforms(C):
    input_size = get_input_size(C.BACKBONE)
    if C.BACKBONE == 'Xception':
        normalize = Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    else:
        # ImageNet mean+std
        normalize = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    return Compose([
        Resize([input_size, input_size]),
        ToTensor(),
        normalize,
    ])

def get_annotation_transforms(C):
    input_size = get_input_size(C.BACKBONE)
    return Compose([
        Resize([input_size, input_size]),
        ToTensor(),
    ])
class ImageDataset(Dataset):
    def __init__(self, data_dir_map, transform=None, target_transform=None,
                 annotations=None, annotation_transform=None,
                reactiontime_filename=None, reactiontime_bridge_filename=None):
        self.filenames = []
        label_values = []
        label_lengths = []
        for data_dir, label in data_dir_map.items():
            dir_filenames = os.listdir(data_dir)
            self.filenames.extend(
                os.path.join(data_dir, dir_filename)
                for dir_filename in dir_filenames
                # filter "._*" filenames
                if not dir_filename.startswith('._')
            )
            label_values.append(label)
            label_lengths.append(len(dir_filenames))
        if annotations is not None:
            # grab the corresponding annotations per example
            annotation_filenames_real = {*os.listdir(annotations)}
            annotation_filenames = []
            for filename in self.filenames:
                basename = os.path.basename(filename)
                assert basename in annotation_filenames_real, (
                    annotations, basename)
                annotation_filenames.append(
                    os.path.join(annotations, basename))
            self.annotation_filenames = annotation_filenames
        else:
            self.annotation_filenames = None
        self.labels = QuickLabels(label_values, label_lengths)
        self.transform = transform
        self.target_transform = target_transform
        self.annotation_transform = annotation_transform
        self.reactiontime_filename = reactiontime_filename
        self.bridge_filename = reactiontime_bridge_filename
        # init the pandas dataframes given the path
        if self.reactiontime_filename and self.bridge_filename:
            self.reaction_times_df = pd.read_csv(self.reactiontime_filename)
            self.bridge_df = pd.read_csv(self.bridge_filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img_path = self.filenames[index]
        image = image_loader(img_path)
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        # regular CYBORG
        if self.annotation_filenames and not self.reactiontime_filename:
            annotation = image_loader(self.annotation_filenames[index])
            if self.annotation_transform:
                annotation = self.annotation_transform(annotation)
                return image, label, annotation
        # CYBORG + reaction times
        elif self.annotation_filenames and self.reactiontime_filename:
            annotation = image_loader(self.annotation_filenames[index])
            if self.annotation_transform:
                annotation = self.annotation_transform(annotation)

            # split off to just filename
            local_img_path = os.path.basename(img_path).split('/')[-1]
            img_ids_row = self.bridge_df.loc[self.bridge_df['img_path'] == local_img_path]
            img_ids = img_ids_row['img_id'].values
            # for each id, load the reaction time in the other set
            reaction_times = np.zeros(len(img_ids))
            for i, img_id in enumerate(img_ids):
                reaction_time_row = self \
                    .reaction_times_df \
                    .loc[self.reaction_times_df['image_id'] == img_id]
                # now, reaction time is the overall time - the time spent annotating
                # this measures the time they spent looking at the image
                # before making a decision as to whether it is real or fake
                reaction_time = reaction_time_row['overall_ann_time'].values \
                    - reaction_time_row['annotation_time'].values 

                # old implementation
                # reaction_time = reaction_time_row['annotation_time'].values

                # very rarely, we get some images things that are sampled twice 
                # choose the max value for now
                reaction_time = reaction_time.max()
                reaction_times[i] = reaction_time
                
            average_reaction_time_per_sample = reaction_times.mean()
            average_reaction_time_per_sample = \
                torch.from_numpy(np.asarray(average_reaction_time_per_sample))
            
            return image, label, annotation, average_reaction_time_per_sample

        elif not self.annotation_filenames and self.reactiontime_filename:
            # reaction time only 
            local_img_path = os.path.basename(img_path).split('/')[-1]
            img_ids_row = self.bridge_df.loc[self.bridge_df['img_path'] == local_img_path]
            img_ids = img_ids_row['img_id'].values
            # for each id, load the reaction time in the other set
            reaction_times = np.zeros(len(img_ids))
            new_rts = np.zeros(len(img_ids))

            for i, img_id in enumerate(img_ids):
                reaction_time_row = self \
                    .reaction_times_df \
                    .loc[self.reaction_times_df['image_id'] == img_id]

                # now, reaction time is the overall time - the time spent annotating
                # this measures the time they spent looking at the image
                # before making a decision as to whether it is real or fake
                reaction_time = reaction_time_row['overall_ann_time'].values \
                    - reaction_time_row['annotation_time'].values 

                # old implementation
                # reaction_time = reaction_time_row['annotation_time'].values

                # very rarely, we get some images things that are sampled twice 
                # choose the max value for now
                reaction_time = reaction_time.max()
                reaction_times[i] = reaction_time

            average_reaction_time_per_sample = reaction_times.mean()
            average_reaction_time_per_sample = \
                torch.from_numpy(np.asarray(average_reaction_time_per_sample))
            return image, label, average_reaction_time_per_sample
        else: 
            # normal cross entropy, no additional annotations
            return image, label

# noinspection PyAbstractClass
class DeepFakesDataModule(LightningDataModule):
    def __init__(self, C, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = C
        self.data_dir_train = os.path.join(C.DATA_DIR_BASE, C.DATA_DIR_TRAIN,
                                           C.TRAIN_SPLIT)
        self.data_dir_val = os.path.join(C.DATA_DIR_BASE, C.DATA_DIR_VAL)
        self.train = self.val = None

    def setup(self, stage=None):
        """called on every GPU"""
        transforms = get_transforms(self.C)
        train_kwargs = {}
        if self.C.LOSS == 'CYBORG' or self.C.LOSS == 'DIFFERENTIABLE_CYBORG+REACTIONTIME':
            logger.info('Data requires human annotations to be loaded.')
            train_kwargs['annotations'] = os.path.join(
                self.C.DATA_DIR_BASE,
                self.C.DATA_DIR_TRAIN,
                self.C.DATA_DIR_TRAIN_ANNOTATIONS,
            )
            train_kwargs['annotation_transform'] = get_annotation_transforms(
                self.C)
        
        if self.C.LOSS == 'CYBORG+REACTIONTIME' or self.C.LOSS == 'REACTIONTIME' \
            or self.C.LOSS == 'DIFFERENTIABLE_REACTIONTIME' or self.C.LOSS == 'DIFFERENTIABLE_CYBORG+REACTIONTIME':
            train_kwargs['reactiontime_filename'] = self.C.REACTIONTIME_FILE
            train_kwargs['reactiontime_bridge_filename'] = self.C.REACTIONTIME_BRIDGE_FILE
            self.train = ImageDataset({
                os.path.join(self.data_dir_train, self.C.DATA_DIR_REAL): 0,
                os.path.join(self.data_dir_train, self.C.DATA_DIR_FAKE): 1,
            }, transform=transforms, **train_kwargs)
        else: 
            self.train = ImageDataset({
                os.path.join(self.data_dir_train, self.C.DATA_DIR_REAL): 0,
                os.path.join(self.data_dir_train, self.C.DATA_DIR_FAKE): 1,
            }, transform=transforms, **train_kwargs)
        self.val = ImageDataset({
            os.path.join(self.data_dir_val, self.C.DATA_DIR_REAL): 0,
            os.path.join(self.data_dir_val, self.C.DATA_DIR_FAKE): 1,
        }, transform=transforms)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.C.BATCH_SIZE,
                          shuffle=True, num_workers=num_cpus(), drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.C.BATCH_SIZE,
                          num_workers=num_cpus(), drop_last=True)


def get_test_data_loaders(C):
    """
    To evaluate accuracy of the models trained under the three scenarios, we
    composed a comprehensive test set of 100,000 synthetically generated images
    from each of six different GAN architectures, ending up with 600,000 total
    test samples. The genuine face datasets used for testing are the FFHQ
    dataset (70,000 images) and the CelebA-HQ dataset (30,000 images), described
    in Sec. 3. For each of the synthetic image sets, the associated real images
    come from the dataset used to train that GAN model. For ProGAN and
    StarGANv2, the real data is CelebA-HQ; for the remaining four StyleGAN sets,
    the real data is FFHQ. This setup aims at demonstrating whether models can
    differentiate between authentic samples and synthetic samples, where the
    latter are generated by a GAN trained on the former.

    JUSTIN: ALL_celebs and ALL_StyleGAN sets combine aspects of the others into several
    a larger dataset at inference times. (i.e. metrics computed across all testing data with each model)

    │   │   ├── 0_real
    │   │   │   ├── celeba-hq_real_aligned
    │   │   │   └── ffhq_aligned
    │   │   └── 1_fake
    │   │       ├── progan_aligned
    │   │       ├── stargan_aligned
    │   │       ├── stylegan1-0.5_aligned
    │   │       ├── stylegan2-0.5_aligned
    │   │       ├── stylegan3-0.5_aligned
    │   │       └── stylegan-ada-0.5_aligned
    """
    data_dir_test = os.path.join(C.DATA_DIR_BASE, C.DATA_DIR_TEST)
    real = os.path.join(data_dir_test, C.DATA_DIR_REAL)
    fake = os.path.join(data_dir_test, C.DATA_DIR_FAKE)

    transforms = get_transforms(C)

    datasets = [
        # For ProGAN and StarGANv2, the real data is CelebA-HQ
        # ImageDataset({
        #     os.path.join(real, 'celeba-hq_real_aligned'): 0,
        #     os.path.join(fake, 'progan_aligned'): 1,
        # }, transform=transforms),
        # ImageDataset({
        #     os.path.join(real, 'celeba-hq_real_aligned'): 0,
        #     os.path.join(fake, 'stargan_aligned'): 1,
        # }, transform=transforms),
        # # For the remaining four StyleGAN sets, the real data is FFHQ
        # ImageDataset(dict_union(
        #     {os.path.join(real, 'ffhq_aligned'): 0},
        #     {os.path.join(fake, 'stylegan1-0.5_aligned', dirname): 1
        #      for dirname in os.listdir(
        #         os.path.join(fake, 'stylegan1-0.5_aligned'))}),
        #     transform=transforms),
        # ImageDataset(dict_union(
        #     {os.path.join(real, 'ffhq_aligned'): 0},
        #     {os.path.join(fake, 'stylegan2-0.5_aligned', dirname): 1
        #      for dirname in os.listdir(
        #         os.path.join(fake, 'stylegan2-0.5_aligned'))}),
        #     transform=transforms),
        # ImageDataset({
        #     os.path.join(real, 'ffhq_aligned'): 0,
        #     os.path.join(fake, 'stylegan3-0.5_aligned'): 1,
        # }, transform=transforms),
        # ImageDataset({
        #     os.path.join(real, 'ffhq_aligned'): 0,
        #     os.path.join(fake, 'stylegan-ada-0.5_aligned'): 1,
        # }, transform=transforms),

        # # all celebs together
        ImageDataset(dict_union(
            {os.path.join(real, 'celeba-hq_real_aligned'): 0},
            {os.path.join(fake, 'progan_aligned'): 1},
            {os.path.join(fake, 'stargan_aligned'): 1}),
            transform=transforms),
        # all the style datasets
        ImageDataset(dict_union(
            {os.path.join(real, 'ffhq_aligned'): 0},
            {os.path.join(fake, 'stylegan1-0.5_aligned', dirname): 1
             for dirname in os.listdir(
                os.path.join(fake, 'stylegan1-0.5_aligned'))},
            {os.path.join(fake, 'stylegan2-0.5_aligned', dirname): 1
             for dirname in os.listdir(
                os.path.join(fake, 'stylegan2-0.5_aligned'))},
            {os.path.join(fake, 'stylegan3-0.5_aligned'): 1},
            {os.path.join(fake, 'stylegan-ada-0.5_aligned'): 1}
            ),
            transform=transforms),

    ]
    # names = ['ProGAN', 'StarGANv2', 'StyleGAN', 'StyleGANv2', 'StyleGANv3',
    #          'StyleGAN2-ADA']
    names = ['celebs_combined', 'StyleGAN_combined']
    if C.QUICK_TEST:
        datasets = datasets[:2]
        names = names[:2]
    # BIST
    print('holy fucking shit these shouldb et ehs same', len(names), len(datasets))

    assert len(names) == len(datasets)
    data_loaders = [DataLoader(ds, batch_size=C.BATCH_SIZE,
                               num_workers=num_cpus(), drop_last=True)
                    for ds in datasets]
    return data_loaders, names
