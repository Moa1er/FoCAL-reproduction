# FoCAL Reproduction

## Setup Instructions

### 1. Create and Activate a Virtual Environment
1. Run the following command to create a virtual environment:
   ```bash
   python -m venv .venv
   ```
2. Activate the virtual environment:
   - On **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

### 2. Install Dependencies
Once the virtual environment is activated, install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Experiment
Run the following command to execute the experiment:
```bash
python -m experiments.rotation_2D
```

### 4. Command-Line Arguments
The `python -m experiments.rotation_2D` script accepts the following arguments:

- **`num_angles`** (int, default: `8`):
  Number of angles for rotation.

- **`dataset`** (str, default: `"cifar10"`):
  Dataset to use for experiments. Options:
  - `"cifar10"`
  - `"cifar100"`
  - `"stl10"`
  - `"imagenet"`

- **`model`** (str, default: `"clip"`):
  Model architecture for downstream classification. Options:
  - `"clip"`
  - `"siglip"`
  - `"resnet"`
  - `"vitb"`
  - `"dino"`
  - `"prlc_r50"`
  - `"prlc_vit"`

- **`logits_model`** (str, default: `"clip"`):
  Model architecture to calculate unsupervised logit energies from for alignment. Options:
  - `"clip"`
  - `"siglip"`
  - `"dino"`

- **`ckpt`** (str, default: `None`):
  Path to the pretrained checkpoint for the model (only used for PRLC models).

- **`N`** (int, default: `-1`):
  Number of samples to process (`-1` for all samples).

- **`diffusion`** (object, default: `DiffusionEnergyArgs()`):
  Diffusion energy arguments.

- **`clip_energy`** (object, default: `CLIPEnergyArgs(factor=1.0)`):
  Classifier energy arguments.

- **`seed`** (int, default: `0`):
  Random seed for reproducibility.

- **`device`** (str, default: `"cuda:0"`):
  Device to run the experiment on (e.g., `"cuda:0"`, `"cpu"`).

- **`stn`** (object, default: `STNArgs(enabled=False, ckpt=None, input_size=32, kernel_size=3)`):
  Optional Spatial Transformer Network (STN) candidate arguments. When enabled, the script adds the STN-transformed image as an extra candidate alongside the standard FoCAL candidates.

  STN sub-arguments:
  - **`stn.enabled`** (bool, default: `False`):
    Whether to add the STN output to the candidate set.

  - **`stn.ckpt`** (str, default: `None`):
    Path to the trained STN checkpoint. Required when `stn.enabled=True`.

  - **`stn.input_size`** (int, default: `32`):
    Input resolution expected by the STN model.

  - **`stn.kernel_size`** (int, default: `3`):
    Kernel size used by the STN localization network.