## Introduction

The codebase provides the training and evaluation code for the submission "AdaInt: Learning Adaptive Intervals for 3D Lookup Tables on Real-time Image Enhancement". It is based on the popular MMEditing toolbox ([v0.11.0](https://github.com/open-mmlab/mmediting/tree/v0.11.0)). Please refer to [ori_README.md](ori_README.md) for the original README.

## Code Structure

- `mmedit/`: the original MMEditing toolbox.
- `adaint/`: the core implementation of the submission, including:
  - `annfiles/`: including the annotation files for FiveK and PPR10K datasets.
  - `dataset.py`: the dataset class for image enhancement (FiveK and PPR10K).
  - `transforms.py`: including some augmentations not provided by MMEditing toolbox.
  - `ailut_transform/`: including the python interfaces and C++ CUDA implementation of the proposed AiLUT-Transform.
  - `model.py`: the implementation of AiLUT model (3D-LUT + AdaInt).
  - `configs/`: including configurations to conduct experiments.

## Prerequisites

### Dependencies

- Ubuntu 18.04.5 LTS
- Python 3.7.10 or higher
- PyTorch 1.8.1 or higher (only verified on 1.8.1)
- **CUDA 10.2**
- **GCC/G++ 7.5**
- **MMCV 1.3.17**
- **MMEditing 0.11.0**

### Installation

You can set up the MMEditing toolbox with conda and pip as follows:

```shell
conda install -c pytorch pytorch=1.8.1 torchvision cudatoolkit -y
pip install -r requirements.txt
pip install -v -e .
```

After setting up the MMEditing toolbox, please complie and install the AiLUT-Transform CUDA extension following the command bellow:

```shell
python adaint/ailut_transform/setup.py install
```

If you fail to complile the CUDA extension, please check the versions of PyTorch, CUDA and GCC/G++ carefully.

## Datasets

The paper use the [FiveK](https://data.csail.mit.edu/graphics/fivek/) and [PPR10K](https://github.com/csjliang/PPR10K) datasets for experiments. It is recommended to refer to the dataset creators first using the above two urls.

### Download

- FiveK

You can download the original FiveK dataset from the dataset [homepage](https://data.csail.mit.edu/graphics/fivek/) and then preprocess the dataset using Adobe Lightroom following the instruction [here](TODO). For fast setting up, you can also download only the 480p dataset preprocessed by Zeng ([[GoogleDrive](https://drive.google.com/drive/folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a?usp=sharing)],[[onedrive](https://connectpolyu-my.sharepoint.com/:f:/g/personal/16901447r_connect_polyu_hk/EqNGuQUKZe9Cv3fPG08OmGEBbHMUXey2aU03E21dFZwJyg?e=QNCMMZ)],[[baiduyun](https://pan.baidu.com/s/1CsQRFsEPZCSjkT3Z1X_B1w):5fyk]), including 8-bit sRGB, 16-bit XYZ input images and 8-bit sRGB groundtruth images.

After downloading the dataset, please unzip the images into the `./data/FiveK` directory. Please also place the annotation files in `./adaint/annfiles/FiveK` to the same directory. The final directory structure is as follows.

```
./data/FiveK
    input/
        JPG/480p/                # 8-bit sRGB inputs
        PNG/480p_16bits_XYZ_WB/  # 16-bit XYZ inputs
    expertC/JPG/480p/            # 8-bit sRGB groundtruths
    train.txt
    test.txt
```

- PPR10K

We download the 360p dataset (`train_val_images_tif_360p` and `masks_360p`) from [PPR10K](https://github.com/csjliang/PPR10K) to conduct our experiments.

After downloading the dataset, please unzip the images into the `./data/PPR10K` directory. Please also place the annotation files in `./adaint/annfiles/PPR10K` to the same directory. The expected directory structure is as follows.

```
data/PPR10K
    source/       # 16-bit sRGB inputs
    source_aug_6/ # 16-bit sRGB inputs with 5 versions of augmented
    masks/        # human-region masks
    target_a/     # 8-bit sRGB groundtruths retouched by expert a
    target_b/     # 8-bit sRGB groundtruths retouched by expert b
    target_c/     # 8-bit sRGB groundtruths retouched by expert c
    train.txt
    train_aug.txt
    test.txt
```

## Usage

### Train on FiveK-sRGB (for photo retouching)

- Configure the experiment by modifying `adaint/configs/fivekrgb.py`. Some critical hyper-parameters:
  - `model.n_ranks`: denoted as `M` in the submission.
  - `model.n_vertices`: denoted as `N` in the submission.
  - `model.en_adaint`: whether to use the AdaInt. If False, the model degenerates to [TPAMI 3D-LUT](https://www4.comp.polyu.edu.hk/~cslzhang/paper/PAMI_LUT.pdf).
  - `model.en_adaint_share`: whether to share AdaInt among color channels (see the `Share-AdaInt` in ablation studies).
  - `model.backbone`: the architecture of backbone (mapping `f` in the submission).

- Start the training using the following command:

```shell
python tools/train.py adaint/configs/fivekrgb.py
```

### Train on FiveK-XYZ (for tone mapping)

- Configure the experiment by modifying `adaint/configs/fivekxyz.py`.

- Start the training using the following command:

```shell
python tools/train.py adaint/configs/fivekxyz.py
```

### Train on PPR10K (for photo retouching)

- Configure the experiment by modifying `adaint/configs/ppr10k.py`.

- Start the training using the following command:

```shell
python tools/train.py adaint/configs/ppr10k.py
```

## License

This project will be released under the [Apache 2.0 license](LICENSE).
