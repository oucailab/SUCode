# Enhancing Underwater Images via Adaptive Semantic-aware Codebook Learning

This repository contains the official implementation of the following paper:
> **Enhancing Underwater Images via Adaptive Semantic-aware Codebook Learning**<br>
> Bosen Lin, Feng Gao<sup>*</sup>, Yanwei Yu, Junyu Dong, Qian Du <br>
> IEEE Transactions on Geoscience and Remote Sensing, 2026<br>

[[Paper](https://ieeexplore.ieee.org/document/11395318)]

## Dependencies and Installation
1. Clone Repo
    ```bash
    git clone https://github.com/oucailab/SUCode.git
    cd SUCode
    ```

2. Create Conda Environment
    ```bash
    conda create -n SUcode python=3.8
    conda activate SUcode
    pip install -r requirements.txt
    python setup.py develop
    ```

## Get Started
### Prepare pretrained models & dataset 

1. You are supposed to download our pretrained model for stage 2 and stage 3 first in the links below and put them in dir `./checkpoints/`:

<table>
<thead>
<tr>
    <th>Model</th>
    <th>:link: Download Links </th>
</tr>
</thead>
<tbody>
<tr>
    <td>SUCode</td>
    <th>[<a href="https://pan.baidu.com/s/17dFgDyLGH-GSajaQOCpDcA?pwd=icpg">Baidu Disk (pwd: icpg)</a>] </th>
</tr>
</tbody>
</table>

2. Unzip UIE dataset and put in dir `./dataset/`.
**The directory structure will be arranged as**:
```
checkpoints
    |- net_stage2_g_best_.pth
    |- net_stage2_d_best_.pth
    |- net_sucode_g_best_.pth
dataset
    |- test
        |- images
            |- ***.jpg
            |- ...
        |- reference
            |- ***.jpg
            |- ...
    |- train
        |- images
            |- ***.jpg
            |- ...
        |- reference
            |- ***.jpg
            |- ...
```

### Training & Testing
Run the following commands for training:

```bash
python basicsr/train.py -opt options/train_SUCode_stage3.yaml
```

Run the following commands for testing:
```bash
python test_sucode.py
```

## Citation
If you find our repo useful for your research, please cite us:
```
@article{lin2026sucode,
  title={Enhancing Underwater Images via Adaptive Semantic-aware Codebook Learning},
  author={Bosen Lin, Feng Gao, Yanwei Yu, Junyu Dong, Qian Du},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.


