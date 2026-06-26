# TRELLIS-Gear

Fine-tuning [TRELLIS by Microsoft](https://github.com/microsoft/TRELLIS) for text-to-3D generation on custom datasets.

---

## Prerequisites

Clone and set up the original TRELLIS repository and follow its environment setup instructions before proceeding.

```bash
git clone https://github.com/microsoft/TRELLIS.git
cd TRELLIS
# Follow environment setup in TRELLIS README
```

---

## Step 1 — Set Up Environment

Run the example script to confirm the environment is working and models are downloading correctly from Hugging Face.

```bash
python example_text.py
```

**To change the model**, edit this line in `example_text.py`:
```python
TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
```
Replace `TRELLIS-text-xlarge` with your desired model variant.

**If you encounter attention errors**, run with explicit settings:
```bash
SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=xformers python example_text.py
```

**To change the generation prompt**, edit the prompt string in `example_text.py`:
```python
outputs = pipeline.run("A chair looking like a avocado.", ...
```

---

## Step 2 — Preparing Dataset

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

---

## Step 3 — Run the Data Pipeline

Follow the dataset preparation instructions in `TRELLIS/DATASET.md` to process your assets into the format required for training.

---

## Step 4 — Train the Model

Run the training commands from the **Training Setup** section of the TRELLIS README.

---

## Step 5 — Swap in Your Trained Weights

Replace the base model checkpoints with your fine-tuned ones. Rename your trained model files to match the base model naming convention so they are picked up automatically.

**Example** — Stage 1 generation model:
```
ss_flow_txt_dit_B_16l8_fp16.json
```

---

## Step 6 — Run Inference

Test your fine-tuned model using `example_text.py` as per Step 1.

---

## Step 7 — Launch the GUI

Start the Gradio interface for interactive inference.

```bash
SPARSE_ATTN_BACKEND=xformers ATTN_BACKEND=xformers python FYP_Gradio.py
```

For the latest version, use the notebook **`test.ipynb`** — it has been updated for Gemma 4 (note: `FYP_Gradio.py` was built for Gemma 3 / Gemini 2.5 flash).

---

## Credits

Built on [TRELLIS](https://github.com/microsoft/TRELLIS) by Microsoft.
