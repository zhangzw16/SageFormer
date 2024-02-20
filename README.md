# SageFormer: Series-Aware Framework for Long-Term Multivariate Time-Series Forecasting

This repository contains the code for the paper "[SageFormer: Series-Aware Framework for Long-Term Multivariate Time-Series Forecasting](https://ieeexplore.ieee.org/abstract/document/10423755)" by Zhenwei Zhang, Linghang Meng, and Yuantao Gu, published in the IEEE Internet of Things Journal.

## Introduction

SageFormer is a novel series-aware graph-enhanced Transformer model designed for long-term forecasting of multivariate time-series (MTS) data. With the proliferation of IoT devices, MTS data has become ubiquitous, necessitating advanced models to forecast future behaviors. SageFormer addresses the challenge of capturing both intra- and inter-series dependencies, enhancing the predictive performance of Transformer-based models.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
To train and evaluate the SageFormer model:

- Clone this repository
- Download datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/share/init?surl=r3KhGd0Q9PJIUZdfEYoymg&pwd=i9iy) and place them in the `./dataset` folder
- Create a virtual environment and activate it
Install requirements pip install -r requirements.txt
- Run scripts in the `./scripts` folder to train and evaluate the model, for example:
    ```bash
    sh scripts/long_term_forecast/ECL_script/SageFormer.sh
    ``` 
- Model checkpoints and logs will be saved to outputs folder

## Contacts
For any questions, please contact the authors at `zzw20 [at] mails.tsinghua.edu.cn`.

## Citation
If you find this code or paper useful for your research, please cite:
```bibtex
@ARTICLE{zhang2024sageformer,
  author={Zhang, Zhenwei and Meng, Linghang and Gu, Yuantao},
  journal={IEEE Internet of Things Journal}, 
  title={SageFormer: Series-Aware Framework for Long-Term Multivariate Time Series Forecasting}, 
  year={2024},
  doi={10.1109/JIOT.2024.3363451}}
```

# Acknowledgement

This library is constructed based on the following repos:
- https://github.com/thuml/Time-Series-Library
- https://github.com/PatchTST/PatchTST
