# AWRaCLe Reproduction with Haar Wavelet DWT Context Enhancement

This repository contains my lab project implementation and reproduction of the paper **AWRaCLe: All-Weather Image Restoration using Visual In-Context Learning**. The original method uses a context pair consisting of a degraded image and its clean counterpart to guide restoration of a query image under adverse weather conditions such as rain, haze, and snow.

As part of this project, I reproduced the original AWRaCLe architecture and added an architectural improvement to the **Degradation Context Extraction (DCE)** block by incorporating **Haar Wavelet Discrete Wavelet Transform (DWT)** features. This additional frequency-aware context branch improved the validation performance compared with the reproduced baseline.

## Project Objective

The goals of this project are:

- Reproduce the AWRaCLe paper for all-weather image restoration.
- Understand how visual in-context learning is used for image restoration.
- Modify the original architecture by adding Haar Wavelet DWT features inside the context extraction block.
- Compare the reproduced AWRaCLe baseline with the improved AWRaCLe + DWT model.

## Base Paper

**AWRaCLe: All-Weather Image Restoration using Visual In-Context Learning**

Authors: Sudarshan Rajagopalan and Vishal M. Patel

Paper: [arXiv:2409.00263](https://arxiv.org/abs/2409.00263)  
Original project page: [AWRaCLe Project](https://sudraj2002.github.io/awraclepage/)  
Original codebase: [github.com/sudraj2002/AWRaCLe](https://github.com/sudraj2002/AWRaCLe)

## Method Overview

AWRaCLe restores a degraded query image using an in-context pair:

- a degraded context image
- the corresponding clean context image

The original architecture extracts CLIP-based visual context features from this pair. These features are processed by **Degradation Context Extraction (DCE)** blocks and fused into the decoder through **Context Fusion (CF)** blocks. This allows the restoration network to use degradation-specific guidance while restoring rain, haze, and snow images.

## My Improvement: Haar Wavelet DWT in DCE Block

In the improved version, I added a Haar Wavelet DWT branch inside the Degradation Context Extraction block.

The added module decomposes each context image into:

- `LL`: low-frequency component
- `LH`, `HL`, `HH`: high-frequency components

The low-frequency component helps capture global degradation patterns such as haze and illumination changes. The high-frequency components help capture local degradation details such as rain streaks, snow particles, edges, and texture changes.

These wavelet features are projected into compact context tokens and concatenated with the original CLIP context tokens before context attention. This makes the context representation both semantic and frequency-aware.

The main added modules are implemented in [`net/model.py`](net/model.py):

- `HaarDWT2D`
- `DWTContextEncoder`
- modified `InContextExtBlock`

## Results

The Haar DWT enhanced model achieved better validation results than the reproduced AWRaCLe baseline.

| Model | Valid Loss | PSNR | SSIM | PSNR Rain | SSIM Rain | PSNR Haze | SSIM Haze | PSNR Snow | SSIM Snow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AWRaCLe | 0.028532 | 27.8587 | 0.8979 | 31.4580 | 0.9188 | 24.5519 | 0.8708 | 27.5664 | 0.9041 |
| AWRaCLe + DWT | **0.025285** | **28.5804** | **0.9035** | **31.7438** | **0.9207** | **26.1830** | **0.8833** | **27.8143** | **0.9066** |

### Improvement Summary

- Validation loss decreased from `0.028532` to `0.025285`.
- Overall PSNR improved by `0.7217 dB`.
- Overall SSIM improved from `0.8979` to `0.9035`.
- The largest improvement was observed for haze restoration, where PSNR improved by `1.6311 dB`.

## Repository Structure

```text
AWRaCLe-main/
|-- assets/                 # Images and figures from the project
|-- awracle_data/           # Dataset folder, if present locally
|-- awracle_data_large/     # Larger dataset folder, if present locally
|-- net/
|   `-- model.py            # AWRaCLe model with Haar DWT enhancement
|-- tools/                  # Dataset/json helper scripts
|-- utils/                  # Dataset, image, metric, and scheduler utilities
|-- options.py              # Training options
|-- train.py                # Training script
|-- test.py                 # Testing/inference script
|-- requirements.txt        # Python dependencies
`-- README.md
```

## Installation

Create a Python environment and install the required packages.

```bash
conda create -n awracle python=3.9 -y
conda activate awracle
pip install -r requirements.txt
```

This project uses PyTorch, PyTorch Lightning, TorchMetrics, CLIP, OpenCV, and image restoration utilities.

## Dataset

The project follows the dataset format used by the original AWRaCLe implementation. The training and validation JSON files should point to degraded images and their corresponding clean target images.

Expected dataset organization is similar to:

```text
data_awracle/
|-- CSD/
|-- Rain13K/
|-- RESIDE/
|-- Snow100k/
|-- Train/
`-- Train_clip/
```

Update the dataset paths in `options.py` or pass the required arguments while training/testing.

## Training

To train the model:

```bash
python train.py
```

or use:

```bash
bash train.sh
```

Important training options are available in [`options.py`](options.py), including dataset paths, batch size, patch size, number of epochs, checkpoint directory, and degradation types.

## Testing

To test a trained checkpoint:

```bash
python test.py --ckpt_name <path_to_checkpoint> --test_dir <path_to_test_data> --test_json <test_json_file>
```

or use:

```bash
bash test.sh
```

The restored outputs are saved to the output directory specified by `--output_path`.

## Custom Inference

For inference on custom degraded images, provide a degraded context image and a clean context image:

```bash
python test.py \
  --mode 2 \
  --test_dir <folder_with_degraded_images> \
  --ckpt_name <path_to_checkpoint> \
  --degrad_context <path_to_degraded_context_image> \
  --clean_context <path_to_clean_context_image> \
  --output_path output/
```

## Key Files Modified for My Contribution

The main architectural change is in [`net/model.py`](net/model.py).

The added Haar DWT branch:

1. Applies fixed Haar filters to context images.
2. Separates low-frequency and high-frequency components.
3. Projects these components into context tokens.
4. Concatenates wavelet tokens with CLIP tokens.
5. Uses the enriched context in the DCE and CF pipeline.

## Acknowledgement

This repository is based on the original AWRaCLe implementation by Sudarshan Rajagopalan and Vishal M. Patel. The original code also builds on PromptIR.

This project was completed as a lab reproduction and improvement study. The main contribution in this repository is the addition of Haar Wavelet DWT based degradation context enhancement and the comparison against the reproduced baseline.

## Citation

If you use the original AWRaCLe work, please cite:

```bibtex
@article{rajagopalan2024awracle,
  title={AWRaCLe: All-Weather Image Restoration using Visual In-Context Learning},
  author={Sudarshan Rajagopalan and Vishal M. Patel},
  journal={arXiv preprint arXiv:2409.00263},
  year={2024}
}
```
