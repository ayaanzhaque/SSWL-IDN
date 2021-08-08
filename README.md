# SSWL-IDN

This repository contains the implementation of the paper ["Window-Level is a Strong Denoising Surrogate"](https://arxiv.org/abs/2105.07153) by Ayaan Haque, Adam Wang, and Abdullah-Al-Zubaer Imran.

We introduce SSWL-IDN, a novel self-supervised CT denoising window-level prediction surrogate task. Our method is task-relevant and related to the downstream task, yielding improved performance over recent methods.

## Abstract

CT image quality is heavily reliant on radiation dose, which causes a trade-off between radiation dose and image quality that affects the subsequent image-based diagnostic performance. However, high radiation can be harmful to both patients and operators. Several (deep learning-based) approaches have been attempted to denoise low dose images. However, those approaches require access to large training sets, specifically the full dose CT images for reference, which can often be difficult to obtain. Self-supervised learning is an emerging alternative for lowering the reference data requirement facilitating unsupervised learning. Currently available self-supervised CT denoising works are either dependent on foreign domains or pretexts that are not very task-relevant. To tackle the aforementioned challenges, we propose a novel self-supervised learning approach, namely Self-Supervised Window-Leveling for Image DeNoising (SSWL-IDN), leveraging an innovative, task-relevant, simple, yet effective surrogate---prediction of the window-leveled equivalent. SSWL-IDN leverages residual learning and a hybrid loss combining perceptual loss and MSE, all incorporated in a VAE framework. Our extensive (in- and cross-domain) experimentation demonstrates the effectiveness of SSWL-IDN in aggressive denoising of CT (abdomen and chest) images acquired at 5% dose level only.

## Model

![Figure](https://github.com/zubaerimran/SSWL-IDN/blob/main/images/model_diagram.jpg?raw=true)

Our model performs CT image denoising using a novel, task-relevant self-supervised surrogate task. We predict window-leveled images from non-window-leveled images as our surrogate task. Window-leveled images can be freely created using the window-leveling parameters in the DICOM metadata. Our model architecture is hybrid between a VAE and RED-CNN. We use the RED-CNN structure with residuals and add in the VAE bottleneck to add randomized noise into the latent code using the reparameterization trick. Our loss function is an intuitive hybrid loss between perceptual loss and MSE loss, encouraging both pixel-wise and feature-wise denoising accuracy.

## Dataset

We primarily collect abdomen scans from the publicly available [Mayo CT](https://www.aapm.org/grandchallenge/lowdosect/) data. For thorough denoising evaluation, we generate the CT scans at 5% dose level using the full dose and quarter dose data (scaling the zero-mean independent noise from 25% to 5% dose level). We use 15 full dose abdomen CT and the corresponding quarter (25%) dose CT scans: 10 scans (1,533 slices, 15% for validation) for training and 5 (633 slices) for testing. 5 chest scans (1,061 slices) are selected from the same library for cross-domain evaluation.

## Results

A brief summary of our results are shown below. Our algorithm SSWL-IDN and our architecture RVAE is compared to various baselines and state-of-the-art methods. In the table, the best fully-supervised scores are bolded, and best semi-supervised scores are underlined.

![Results](https://github.com/zubaerimran/SSWL-IDN/blob/main/images/archi-table.png?raw=true)

![Results](https://github.com/zubaerimran/SSWL-IDN/blob/main/images/ssl-table.png?raw=true)

![Results](https://github.com/zubaerimran/SSWL-IDN/blob/main/images/roi-preds.png?raw=true)

## Code

The code has been written in Python using the Pytorch framework. Training requries a GPU. We provide a Jupyter Notebook, which can be run in Google Colab, containing the algorithm in a usable version. Open [`SSWL-IDN.ipynb`](https://github.com/ayaanzhaque/MultiMix/blob/main/MultiMix.ipynb) and run it through. Uncomment the training cell to train the model. We additionally provide python scripts.

## Citation

If you find this repo or the paper useful, please cite:

```
@article{haque2021window,
  title={Window-Level is a Strong Denoising Surrogate},
  author={Haque, Ayaan and Wang, Adam and Imran, Abdullah-Al-Zubaer},
  journal={arXiv preprint arXiv:2105.07153},
  year={2021}
}
```
