# DAVIS Speech Separation with Muddy Mix Dataset

This project uses the DAVIS framework for speech separation on the Muddy Mix dataset. DAVIS is a Diffusion-based Audio-VIsual Separation framework that leverages generative learning to separate sounds from audio mixtures, conditioned on visual information.

## Overview

DAVIS employs a generative diffusion model and a Separation U-Net to synthesize separated sounds directly from Gaussian noise. This approach is particularly effective for high-quality sound separation across diverse categories.

In this implementation, we adapt DAVIS for speech separation using the Muddy Mix dataset, which contains audio mixtures for speech separation tasks.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/beautifulchoi/DAVIS-speech-sep.git
cd DAVIS-speech-sep
```

2. Create a conda environment:
```bash
conda create --name davis-speech python=3.11
conda activate davis-speech
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The Muddy Mix dataset is used for training and evaluation. The dataset files are located in the `data/` directory with CSV files for training, validation, and testing splits.

To create the CSV files, we filtered the dataset by comparing the maximum volume of the speech components to the overall mix. Only videos containing speech with a volume above a certain threshold were included in the valid dataset for training.

## Training

To train the model:

1. Modify the dataset paths in the configuration files.
2. Run the training script:
```bash
python main_fm_muddy.py  # or appropriate training script
```

## Evaluation

To evaluate the model:

1. Set the evaluation mode in the script.
2. Run:
```bash
python main_fm_muddy.py --mode eval --split test
```

## Results

The model achieves high-quality speech separation on the Muddy Mix dataset, demonstrating the effectiveness of the generative approach for audio-visual separation tasks.

## Citation

If you use this code, please cite the original DAVIS paper:

```
@InProceedings{Huang_2024_ACCV,
    author    = {Huang, Chao and Liang, Susan and Tian, Yapeng and Kumar, Anurag and Xu, Chenliang},
    title     = {High-Quality Visually-Guided Sound Separation from Diverse Categories},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2024},
    pages     = {35-49}
}
```
