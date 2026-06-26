# TRELLIS-Gear

Fine-tuning [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft for text-to-3D gear generation.

---

## Prerequisites

Clone and set up the original TRELLIS repository and follow its environment setup instructions before proceeding.

```bash
git clone https://github.com/microsoft/TRELLIS.git
cd TRELLIS
# Follow environment setup in TRELLIS README
```

---

## How to Run (Using the Fine-Tuned Model)
 
The fine-tuned model (trained on gear datasets) is available at:
**[Hugging Face]([https://huggingface.co/Celesteang/TRELLIS-Gear/tree/main](https://huggingface.co/Celeste1812/TRELLIS-Gear/blob/main/README.md))**

This model was finetuned on the TRELLIS-text-base version of the TRELLIS model.
 
### Step 1 — Verify Your Environment
 
Run the example script to confirm the environment is working and models are downloading correctly from Hugging Face.

```bash
python example_text.py
```

**To change the model**, edit this line in `example_text.py`:
```python
TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
```
Replace `TRELLIS-text-xlarge` with `TRELLIS-text-base`.

**If you encounter attention errors**, run with explicit settings:
```bash
SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=xformers python example_text.py
```

**To change the generation prompt**, edit the prompt string in `example_text.py`:
```python
outputs = pipeline.run("A chair looking like a avocado.", ...
```
 
### Step 2 — Swap in the Fine-Tuned Weights
 
Download the weights from Hugging Face and replace the base model checkpoints. 

 
### Step 3 — Run Inference
 
For the latest version of model, use the notebook **`test.ipynb`** instead — it has been updated for Gemma 4.

 
### Step 4 — Launch the GUI
 
Start the Gradio interface for interactive inference.
 
```bash
SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=xformers python FYP_Gradio.py
```
 
---

## How to Train / Fine-Tune Model
 
### Step 1 — Prepare Your Dataset

Add your 3D assets and captions to the TRELLIS directory.

```
TRELLIS/
├── your_dataset/
│   ├── Gear_00000.stl
│   ├── Gear_00001.stl
│   └── ...
└── captions.csv 
```

`captions.csv` must have the following headers:
 
| `sha256` | `filename` | `local_path` | `aesthetic` | `captions` |
|----------|------------|--------------|-------------|------------|
| file hash | Gear_00000.stl | your_dataset/Gear_00000.stl | 9 | "The gear is a cylindrical component with..." |
| file hash | Gear_00001.sil | your_dataset/Gear_00001.stl | 9 | "The gear has 6 modules and 28 teeth..." |


### Step 2 — Run the Data Pipeline

Follow the dataset preparation instructions in `TRELLIS/DATASET.md` to process your assets into the format required for training.


### Step 3 — Train the Model

Run the training commands from the **Training Setup** section of the TRELLIS README.

#### Fine-tuning from a checkpoint (instead of training from scratch)
 
Edit `TRELLIS/trellis/trainers/base.py` and set `finetune_ckpt` in the `Trainer` class:
 
```python
finetune_ckpt={"denoiser": "/path/to/your/checkpoint/checkpoint.pt"},
```
 
**Note:** `"denoiser"` is the key for `ss_flow_txt_dit_B_16l8_fp16` — other model configs may use a different key name. To find the correct key for your model, check the first table under [Training Setup](https://github.com/microsoft/TRELLIS/tree/main#training-setup) in the TRELLIS README.


### Step 4 — Swap in Your Trained Weights

Replace the base model checkpoints with your fine-tuned ones. Rename your trained model files to match the base model naming convention so they are picked up automatically. The 

**Example** — Stage 1 generation model:
```
ss_flow_txt_dit_B_16l8_fp16.safetensors
```

**Note**: You have to convert the training checkpoint to the safetensor version for inference before replacing base model checkpoints.

---

## Credits

Built on [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft.
Dataset credit: Sun, Yuewan; Li, Xingang; Sha, Zhenghui (2024). [https://doi.org/10.18738/T8/KV7HON](https://doi.org/10.18738/T8/KV7HON)
