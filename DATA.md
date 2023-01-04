# Data

Path on CRC: `/scratch365/aboyd3/DataSynFace/ALL_IMAGES/`

Contains:

- All training/val/test data
- All human annotations and the x6/x7 datasets
- Validation sets in val/
- Test sets in test/

Data directory tree

```
├── ALL_IMAGES
│   ├── test
│   │   ├── 0_real
│   │   │   ├── celeba-hq_real_aligned
│   │   │   └── ffhq_aligned
│   │   └── 1_fake
│   │       ├── progan_aligned
│   │       ├── stargan_aligned
│   │       ├── stylegan1-0.5_aligned
│   │       │   ├── 092000
│   │       │   ├── 093000
                    ...
│   │       │   ├── 098000
│   │       │   └── 099000
│   │       ├── stylegan2-0.5_aligned
│   │       │   ├── 092000
│   │       │   ├── 093000
                    ...
│   │       │   ├── 098000
│   │       │   └── 099000
│   │       ├── stylegan3-0.5_aligned
│   │       └── stylegan-ada-0.5_aligned
│   ├── train
│   │   ├── 6x_data
│   │   │   ├── 0_real
│   │   │   └── 1_fake
│   │   ├── 7x_data
│   │   │   ├── 0_real
│   │   │   └── 1_fake
│   │   ├── heatmaps
│   │   └── original_data
│   │       ├── 0_real
│   │       └── 1_fake
│   └── val
│       ├── 0_real
│       └── 1_fake
├── AugmentedImages
│   ├── aligned
│   │   ├── 2
│   │   ├── 4
            ...
│   │   ├── SG2
│   │   └── SREFI
│   ├── cropped
│   │   ├── 12_cropped
│   │   ├── 14_cropped
            ...
│   │   ├── 8_cropped
│   │   └── Original_cropped
│   └── unedited
│       ├── 4
│       ├── 6
        ...
│       ├── SG2
│       └── SREFI
├── BlurredImages
│   ├── aligned
│   │   ├── heatmaps
│   │   ├── ND-Real
                    ...
│   │   ├── SREFI
│   │   └── vfhq_val
│   │       ├── sequence_yval0_fUCGTgkV0ESHa1jFiuIg3inTg__rK6nCavnFvM_3244_3391
│   │       ├── sequence_yval0_rUCGTgkV0ESHa1jFiuIg3inTg__0pZmF5Ny2hM
            ...
│   │       ├── sequence_yval2_fUCSK2DOrVY3-ntcjVMGe8t3A__9TCFsLNo6LU_4732_5488
│   │       └── sequence_yval2_rUCSK2DOrVY3-ntcjVMGe8t3A__9TCFsLNo6LU
│   ├── cropped
│   │   ├── ND-Real
│   │   ├── SG2
│   │   └── SREFI
│   └── unedited
│       ├── 999
│       ├── 999_cropped
                    ...
│       ├── sg2aidan0.5
│       ├── SG2_web
│       │   ├── synthface_data
│       │   └── synthface_meta
│       └── SREFI
├── DFFD
│   ├── eval
│   │   ├── Fake
│   │   └── Real
│   ├── large_data_3x
│   │   ├── train
│   │   │   └── face
│   │   │       ├── 0_real
│   │   │       └── 1_fake
│   │   └── val
│   │       └── face
│   │           ├── 0_real
│   │           └── 1_fake
│   ├── large_data_6x
│   │   ├── train
│   │   │   └── face
│   │   │       ├── 0_real
│   │   │       └── 1_fake
│   │   └── val
│   │       └── face
│   │           ├── 0_real
│   │           └── 1_fake
│   ├── large_data_6x_msu
│   │   ├── eval
│   │   │   ├── Fake
│   │   │   └── Real
│   │   └── train
│   │       ├── Fake
│   │       └── Real
│   ├── large_data_7x
│   │   ├── train
│   │   │   └── face
│   │   │       ├── 0_real
│   │   │       └── 1_fake
│   │   └── val
│   │       └── face
│   │           ├── 0_real
│   │           └── 1_fake
│   ├── large_data_7x_msu
│   │   ├── eval
│   │   │   ├── Fake
│   │   │   └── Real
│   │   └── train
│   │       ├── Fake
│   │       └── Real
│   ├── Mask
│   ├── test
│   │   ├── Fake
│   │   └── Real
│   ├── test_pg
│   │   └── face
│   │       ├── 0_real
│   │       └── 1_fake
│   ├── test_sg
│   │   └── face
│   │       ├── 0_real
│   │       └── 1_fake
│   ├── test_sg2
│   │   └── face
│   │       ├── 0_real
│   │       └── 1_fake
│   ├── test_sg3
│   │   └── face
│   │       ├── 0_real
│   │       └── 1_fake
│   ├── test_sg-ada
│   │   └── face
│   │       ├── 0_real
│   │       └── 1_fake
│   ├── test_stgn
│   │   └── face
│   │       ├── 0_real
│   │       ├── 1_fake
│   │       └── 1_fake_old
│   ├── train
│   │   ├── Fake
│   │   └── Real
│   ├── train_largedata
│   │   ├── Fake
│   │   └── Real
│   └── val_largedata
│       ├── Fake
│       └── Real
├── images
├── raw_heatmaps
├── StyleGAN
│   ├── testing_data
│   │   ├── synthface_data
│   │   └── synthface_meta
│   └── validation_data
│       ├── synthface_data
│       └── synthface_meta
└── TestData
    ├── CelebA
    │   ├── celeba-hq_real_aligned
    │   ├── celeba_real_aligned
    │   └── data1024x1024
    ├── FaceForensics
    │   ├── aligned
    │   │   ├── Deepfakes
    │   │   │   ├── 992_980
    │   │   │   ├── 993_989
                    ...
    │   │   │   ├── 998_561
    │   │   │   └── 999_960
    │   │   ├── Face2Face
    │   │   │   ├── 992_980
    │   │   │   ├── 993_989
                    ...
    │   │   │   ├── 998_561
    │   │   │   └── 999_960
    │   │   ├── FaceShifter
    │   │   │   ├── 992_980
    │   │   │   ├── 993_989
                    ...
    │   │   │   ├── 998_561
    │   │   │   └── 999_960
    │   │   ├── FaceSwap
    │   │   │   ├── 992_980
    │   │   │   ├── 993_989
                    ...
    │   │   │   ├── 998_561
    │   │   │   └── 999_960
    │   │   ├── NeuralTextures
    │   │   │   ├── 992_980
    │   │   │   ├── 993_989
                    ...
    │   │   │   ├── 998_561
    │   │   │   └── 999_960
    │   │   └── youtube
    │   │       ├── 992
    │   │       ├── 993
                ...
    │   │       ├── 998
    │   │       └── 999
    │   ├── manipulated_sequences
    │   │   ├── Deepfakes
    │   │   │   ├── masks
    │   │   │   │   └── images
    │   │   │   │       ├── 992_980
    │   │   │   │       ├── 993_989
                        ...
    │   │   │   │       ├── 998_561
    │   │   │   │       └── 999_960
    │   │   │   └── raw
    │   │   │       └── images
    │   │   │           ├── 992_980
    │   │   │           ├── 993_989
                    ...
    │   │   │           ├── 998_561
    │   │   │           └── 999_960
    │   │   ├── Face2Face
    │   │   │   ├── masks
    │   │   │   │   └── images
    │   │   │   │       ├── 992_980
    │   │   │   │       ├── 993_989
                        ...
    │   │   │   │       ├── 998_561
    │   │   │   │       └── 999_960
    │   │   │   └── raw
    │   │   │       └── images
    │   │   │           ├── 992_980
    │   │   │           ├── 993_989
                    ...
    │   │   │           ├── 998_561
    │   │   │           └── 999_960
    │   │   ├── FaceShifter
    │   │   │   └── raw
    │   │   │       └── images
    │   │   │           ├── 992_980
    │   │   │           ├── 993_989
                    ...
    │   │   │           ├── 998_561
    │   │   │           └── 999_960
    │   │   ├── FaceSwap
    │   │   │   ├── masks
    │   │   │   │   └── images
    │   │   │   │       ├── 992_980
    │   │   │   │       ├── 993_989
                        ...
    │   │   │   │       ├── 998_561
    │   │   │   │       └── 999_960
    │   │   │   └── raw
    │   │   │       └── images
    │   │   │           ├── 992_980
    │   │   │           ├── 993_989
                    ...
    │   │   │           ├── 998_561
    │   │   │           └── 999_960
    │   │   └── NeuralTextures
    │   │       ├── masks
    │   │       │   └── images
    │   │       │       ├── 992_980
    │   │       │       ├── 993_989
                        ...
    │   │       │       ├── 998_561
    │   │       │       └── 999_960
    │   │       └── raw
    │   │           └── images
    │   │               ├── 992_980
    │   │               ├── 993_989
                ...
    │   │               ├── 998_561
    │   │               └── 999_960
    │   └── original_sequences
    │       ├── actors
    │       │   └── raw
    │       │       ├── images
    │       │       │   ├── 27__walking_down_indoor_hall_disgust
    │       │       │   ├── 28__exit_phone_room
                            ...
    │       │       │   ├── 28__walking_down_street_outside_angry
    │       │       │   └── 28__walking_outside_cafe_disgusted
    │       │       └── videos
    │       └── youtube
    │           └── raw
    │               └── images
    │                   ├── 992
    │                   ├── 993
            ...
    │                   ├── 998
    │                   └── 999
    ├── FFHQ
    │   ├── ffhq_aligned
    │   └── FFHQ-Original
    ├── MSU_GANs
    │   ├── images
    │   │   ├── STARGAN_2 [error opening dir]
    │   │   ├── STGAN [error opening dir]
                ...
    │   │   ├── WGAN [error opening dir]
    │   │   └── WGANGP [error opening dir]
    │   └── msu_aligned
    │       ├── STARGAN_2
    │       ├── STGAN
            ...
    │       ├── WGAN
    │       └── WGANGP
    ├── ProGAN
    │   ├── progan
    │   └── progan_aligned
    ├── SG2_test
    │   ├── 0_real
    │   └── 1_fake
    ├── StarGANv2
    │   ├── stargan
    │   ├── stargan_aligned
    │   └── stargan_aligned_old
    ├── StyleGAN
    │   ├── stylegan1-0.5
    │   │   ├── 092000
    │   │   ├── 093000
                ...
    │   │   ├── 098000
    │   │   └── 099000
    │   ├── stylegan1-0.5_aligned
    │   │   ├── 092000
    │   │   ├── 093000
                ...
    │   │   ├── 098000
    │   │   └── 099000
    │   ├── stylegan1-1.0
    │   │   ├── 092000
    │   │   ├── 093000
                ...
    │   │   ├── 098000
    │   │   └── 099000
    │   └── stylegan1-1.0_aligned
    │       ├── 092000
    │       ├── 093000
            ...
    │       ├── 098000
    │       └── 099000
    ├── StyleGAN2
    │   ├── stylegan2-0.5
    │   │   ├── 092000
    │   │   ├── 093000
                ...
    │   │   ├── 098000
    │   │   └── 099000
    │   ├── stylegan2-0.5_aligned
    │   │   ├── 092000
    │   │   ├── 093000
                ...
    │   │   ├── 098000
    │   │   └── 099000
    │   ├── stylegan2-1.0
    │   │   ├── 092000
    │   │   ├── 093000
                ...
    │   │   ├── 098000
    │   │   └── 099000
    │   └── stylegan2-1.0_aligned
    │       ├── 092000
    │       ├── 093000
            ...
    │       ├── 098000
    │       └── 099000
    ├── StyleGAN3
    │   ├── stylegan3-0.5
    │   └── stylegan3-0.5_aligned
    ├── StyleGAN-ADA
    │   ├── 1
    │   ├── 1_aligned
    │   ├── stylegan-ada-0.5
    │   └── stylegan-ada-0.5_aligned
    ├── test_stgn
    │   └── face
    │       ├── 0_real
    │       ├── 1_fake
    │       └── 1_fake_aligned
    └── vfhq
        ├── VideoForensicsHQ
        │   ├── test
        │   │   ├── fake
        │   │   │   ├── subset_captured
        │   │   │   │   ├── identity_ctst0
        │   │   │   │   └── identity_ctst1
        │   │   │   ├── subset_ravdess
        │   │   │   │   ├── identity_rtst01
        │   │   │   │   ├── identity_rtst07
        │   │   │   │   ├── identity_rtst15
        │   │   │   │   └── identity_rtst23
        │   │   │   └── subset_youtube
        │   │   │       ├── identity_ytst0
        │   │   │       ├── identity_ytst1
        │   │   │       ├── identity_ytst2
        │   │   │       └── identity_ytst3
        │   │   └── real
        │   │       ├── subset_captured
        │   │       │   ├── identity_ctst0
        │   │       │   └── identity_ctst1
        │   │       ├── subset_ravdess
        │   │       │   ├── identity_rtst01
        │   │       │   ├── identity_rtst07
        │   │       │   ├── identity_rtst15
        │   │       │   └── identity_rtst23
        │   │       └── subset_youtube
        │   │           ├── identity_ytst0
        │   │           ├── identity_ytst1
        │   │           ├── identity_ytst2
        │   │           └── identity_ytst3
        │   ├── training
        │   │   ├── fake
        │   │   │   ├── subset_captured
        │   │   │   │   ├── identity_ctrn0
        │   │   │   │   ├── identity_ctrn1
        │   │   │   │   └── identity_ctrn2
        │   │   │   ├── subset_ravdess
        │   │   │   │   ├── identity_rtrn14
        │   │   │   │   ├── identity_rtrn16
                            ...
        │   │   │   │   ├── identity_rtrn22
        │   │   │   │   └── identity_rtrn24
        │   │   │   └── subset_youtube
        │   │   │       ├── identity_ytrn0
        │   │   │       ├── identity_ytrn1
        │   │   │       ├── identity_ytrn2
        │   │   │       ├── identity_ytrn3
        │   │   │       ├── identity_ytrn4
        │   │   │       ├── identity_ytrn5
        │   │   │       └── identity_ytrn6
        │   │   └── real
        │   │       ├── subset_captured
        │   │       │   ├── identity_ctrn0
        │   │       │   ├── identity_ctrn1
        │   │       │   └── identity_ctrn2
        │   │       ├── subset_ravdess
        │   │       │   ├── identity_rtrn14
        │   │       │   ├── identity_rtrn16
                            ...
        │   │       │   ├── identity_rtrn22
        │   │       │   └── identity_rtrn24
        │   │       └── subset_youtube
        │   │           ├── identity_ytrn0
        │   │           ├── identity_ytrn1
        │   │           ├── identity_ytrn2
        │   │           ├── identity_ytrn3
        │   │           ├── identity_ytrn4
        │   │           ├── identity_ytrn5
        │   │           └── identity_ytrn6
        │   └── validation
        │       ├── fake
        │       │   ├── subset_captured
        │       │   │   ├── identity_cval0
        │       │   │   └── identity_cval1
        │       │   ├── subset_ravdess
        │       │   │   ├── identity_rval02
        │       │   │   ├── identity_rval09
        │       │   │   ├── identity_rval12
        │       │   │   └── identity_rval20
        │       │   └── subset_youtube
        │       │       ├── identity_yval0
        │       │       ├── identity_yval1
        │       │       └── identity_yval2
        │       └── real
        │           ├── subset_captured
        │           │   ├── identity_cval0
        │           │   └── identity_cval1
        │           ├── subset_ravdess
        │           │   ├── identity_rval02
        │           │   ├── identity_rval09
        │           │   ├── identity_rval12
        │           │   └── identity_rval20
        │           └── subset_youtube
        │               ├── identity_yval0
        │               ├── identity_yval1
        │               └── identity_yval2
        └── VideoForensicsHQ_Coarse
            ├── test
            │   ├── images
            │   │   ├── sequence_ytst3_fUCST7F89qN0XMV9QhTWEom8A__GWC-MWL3-rc_2603_2902
            │   │   ├── sequence_ytst3_fUCST7F89qN0XMV9QhTWEom8A__redHatsDXho_8225_9548
                        ...
            │   │   ├── sequence_ytst3_rUCST7F89qN0XMV9QhTWEom8A__redHatsDXho
            │   │   └── sequence_ytst3_rUCST7F89qN0XMV9QhTWEom8A__TOjhY-BAwJM
            │   ├── images_aligned
            │   │   ├── sequence_ytst3_fUCST7F89qN0XMV9QhTWEom8A__GWC-MWL3-rc_2603_2902
            │   │   ├── sequence_ytst3_fUCST7F89qN0XMV9QhTWEom8A__redHatsDXho_8225_9548
                        ...
            │   │   ├── sequence_ytst3_rUCST7F89qN0XMV9QhTWEom8A__redHatsDXho
            │   │   └── sequence_ytst3_rUCST7F89qN0XMV9QhTWEom8A__TOjhY-BAwJM
            │   └── videos
            ├── training
            │   └── videos
            └── validation
                ├── images
                │   ├── sequence_yval0_fUCGTgkV0ESHa1jFiuIg3inTg__rK6nCavnFvM_3244_3391
                │   ├── sequence_yval0_rUCGTgkV0ESHa1jFiuIg3inTg__0pZmF5Ny2hM
                        ...
                │   ├── sequence_yval2_fUCSK2DOrVY3-ntcjVMGe8t3A__9TCFsLNo6LU_4732_5488
                │   └── sequence_yval2_rUCSK2DOrVY3-ntcjVMGe8t3A__9TCFsLNo6LU
                └── images_aligned
                    ├── sequence_yval0_fUCGTgkV0ESHa1jFiuIg3inTg__rK6nCavnFvM_3244_3391
                    ├── sequence_yval0_rUCGTgkV0ESHa1jFiuIg3inTg__0pZmF5Ny2hM
                        ...
                    ├── sequence_yval2_fUCSK2DOrVY3-ntcjVMGe8t3A__9TCFsLNo6LU_4732_5488
                    └── sequence_yval2_rUCSK2DOrVY3-ntcjVMGe8t3A__9TCFsLNo6LU
```

Generated after using `tree -d` on data dir with code

```python
import re
from collections import OrderedDict

with open('all_data_dirs_tree.txt', 'r') as f:
    lines = f.read().splitlines()
dir_map = OrderedDict()
parent_map = {-1: None}
max_print_at_depth = 8
k = 2
prev_depth = -1
depth_count = 0
out_lines = []
for i, line in enumerate(lines):
    match = re.match(r'^(.*)(?:├──|└──) (.+)$', line)
    if match is None:
        continue
    pre, dirname = match.groups()
    depth = len(pre) // 4
    parent_map[depth] = dirname
    parent_key = parent_map[depth - 1]
    dir_map[parent_key] = dir_map.get(parent_key, []) + [dirname]
    if (i + 1) == len(lines):
        i += 1
        depth -= 1
    if depth != prev_depth or (i + 1) == len(lines):
        if depth_count < max_print_at_depth:
            out_lines.extend(lines[i - depth_count:i])
        else:
            out_lines.extend(
                lines[i - max_print_at_depth:i - max_print_at_depth + k])
            out_lines.append((depth + 2) * 4 * ' ' + '...')
            out_lines.extend(lines[i - k:i])
        depth_count = 0
    depth_count += 1
    prev_depth = depth
for out_line in out_lines:
    print(out_line)
```