<!-- README: T1D Data-Based Digital Twin Workshop (HTML-only) -->

<div style="background-color:#0f1115; padding:28px; border-radius:14px; display:table; width:100%; border:1px solid #23262e; box-shadow:0 6px 18px rgba(0,0,0,0.25);">

  <div style="display:table-cell; vertical-align:top; width:70%; padding-right:16px;">
    <h1 style="color:#4FC3F7; margin:0 0 6px 0; font-size:34px; line-height:1.2;">
      T1D Data-Based Digital Twin • Workshop
    </h1>
    <p style="font-size:18px; color:#cfd8dc; margin:0;">
      Hands-On training on T1D simulators, data pipelines, and personalization methods for individualized care
    </p>
    <p style="font-size:14px; color:#90a4ae; margin:8px 0 0 0;">
      Repo: <code>t1d-data-based-digital-twin</code>
    </p>
  </div>

  <div style="display:table-cell; vertical-align:middle; text-align:left; width:30%; padding-right:20px;">
    <img src="https://micelab.udg.edu/wp-content/uploads/2022/08/MICElab-letras_png-300x119.png" alt="MICELab" style="height:64px; border-radius:8px;">
  </div>
</div>

<div style="margin-top:18px; display:flex; gap:10px; flex-wrap:wrap;">
  <a href="#notebooks" style="text-decoration:none; background:#1e1e1e; color:#e0f7fa; padding:10px 14px; border-radius:8px; border:1px solid #2c2c2c;">Notebooks</a>
  <a href="#quickstart" style="text-decoration:none; background:#1e1e1e; color:#e0f7fa; padding:10px 14px; border-radius:8px; border:1px solid #2c2c2c;">Quickstart</a>
  <a href="#data-layout" style="text-decoration:none; background:#1e1e1e; color:#e0f7fa; padding:10px 14px; border-radius:8px; border:1px solid #2c2c2c;">Data</a>
  <a href="#structure" style="text-decoration:none; background:#1e1e1e; color:#e0f7fa; padding:10px 14px; border-radius:8px; border:1px solid #2c2c2c;">Structure</a>
  <a href="LICENSE.md" style="text-decoration:none; background:#1e1e1e; color:#e0f7fa; padding:10px 14px; border-radius:8px; border:1px solid #2c2c2c;">License</a>
</div>

<hr style="border:none; border-top:1px solid #2c2c2c; margin:18px 0;">

<h2 id="notebooks" style="color:#4FC3F7;">Notebooks</h2>

<div style="display:flex; gap:16px; flex-wrap:wrap;">

  <div style="flex:1 1 360px; background:#111317; border:1px solid #23262e; border-radius:12px; padding:16px;">
    <h3 style="margin-top:0; color:#80deea;">Hands-On Part A: Data & Simulator Basics</h3>
    <p style="color:#cfd8dc; line-height:1.6;">
      Load and inspect the pre-trained model and dataset, apply preprocessing and save for later.
      You will build the foundation for downstream personalization.
    </p>
    <p style="margin:10px 0 0 0;">
      <a href="workshops/hands-on-part-a.ipynb" style="background:#1e88e5; color:#fff; padding:8px 12px; border-radius:8px; text-decoration:none;">Open Part A</a>
    </p>
  </div>

  <div style="flex:1 1 360px; background:#111317; border:1px solid #23262e; border-radius:12px; padding:16px;">
    <h3 style="margin-top:0; color:#80deea;">Hands-On Part B: Personalization Methods</h3>
    <p style="color:#cfd8dc; line-height:1.6;">
      Implement and evaluate personalization strategies using the provided GAN-based components and utility modules. Fine-tune the pre-trained model and simulate data.
    </p>
    <p style="margin:10px 0 0 0;">
      <a href="workshops/hands-on-part-b.ipynb" style="background:#1e88e5; color:#fff; padding:8px 12px; border-radius:8px; text-decoration:none;">Open Part B</a>
    </p>
  </div>

</div>

<hr style="border:none; border-top:1px solid #2c2c2c; margin:18px 0;">


<h2 id="quickstart" style="color:#4FC3F7;">Quickstart</h2>

## Local Setup (Recommended)

### 1. (Optional) Install VS Code
Download and install [Visual Studio Code](https://code.visualstudio.com/).

---

### 2. Clone the repository

#### Windows (PowerShell)
```powershell
cd path\to\where\you\want\the\project
git clone https://github.com/oriolbustos/t1d-data-based-digital-twin
cd t1d-data-based-digital-twin
```

#### Mac/Linux (bash)
```bash
cd /path/to/where/you/want/the/project
git clone https://github.com/oriolbustos/t1d-data-based-digital-twin
cd t1d-data-based-digital-twin
```

---

### 3. Install Python 3.10
Make sure to be running without any venv or conda environment activated.

#### Windows (PowerShell)
```powershell
.\install-python310.ps1
```
Be ready to give permissions when the shield icon appears on your taskbar! Won't continue until you give the administrator account of your PC permission.

#### Mac/Linux (bash)
Make sure you have Python 3.10 installed. On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv
```

On macOS (using [Homebrew](https://brew.sh/)):
```bash
brew install python@3.10
```

---

If the installation run without any errors and it printed "Python 3.10" or a variant of that, the installation was correct.

### 4. Create a virtual environment

#### Windows (PowerShell)
```powershell
python3.10 -m venv .venv
```

#### Mac/Linux (bash)
```bash
python3.10 -m venv .venv
```

Activate it:

- **Windows (PowerShell)**:
```powershell
.\.venv\Scripts\Activate
```

- **Mac/Linux (bash)**:
```bash
source .venv/bin/activate
```

---

### 5. Open in VS Code
```bash
code .
```

- Select the **Python 3.10 virtual environment** you just created as the kernel.
- Install ipykernel when prompted to do (will be done automatically usually).
- Run the notebooks/scripts on the `workshops` folders from there.

---

## Quickstart on Google Colab (Version Problems May Occur)

1. In your Google Drive, create a folder named:
   ```
   t1d-data-based-digital-twin
   ```

2. Upload the repository folders inside this folder, **excluding any hidden/system folders** (like `.git`, `.venv`, etc.).

3. Open the notebooks inside the `workshop-colab` folder (they are adapted versions of the local `workshops` notebooks).

4. When running setup, **grant access to your Google Drive** when prompted.

5. ⚠️ **Important**: Change the system path in the notebook to match your own Google Drive folder hierarchy.
   Example:
   ```python
   import sys
   sys.path.append('/content/drive/MyDrive/t1d-data-based-digital-twin')
   ```

---

<hr style="border:none; border-top:1px solid #2c2c2c; margin:18px 0;">

<h2 id="data-layout" style="color:#4FC3F7;">Data</h2>
<ul style="color:#cfd8dc; line-height:1.8;">
  <li><code>data/data_filtered.csv</code> filtered dataset for rapid experimentation</li>
</ul>

<hr style="border:none; border-top:1px solid #2c2c2c; margin:18px 0;">

<h2 id="structure" style="color:#4FC3F7;">Repository Structure</h2>

<pre style="background:#0b0d10; color:#e3f2fd; padding:14px; border-radius:10px; border:1px solid #23262e; overflow:auto;">
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── clip_constraint.py
│   ├── data_classes.py
│   ├── discriminator.py
│   ├── gan.py
│   ├── latent_space.py
│   ├── loss_functions.py
│   ├── mini_batch_discriminator.py
│   ├── simulate.py
│   └── utils.py
├── data/
│   ├── data_filtered.csv
│   └── data_processed.csv
├── misc/
│   ├── bg_scalers.joblib
│   ├── data_clean.ipynb
│   ├── generator_model_*.h5
│   └── model.png
└── workshops/
    ├── hands-on-part-a.ipynb
    └── hands-on-part-b.ipynb
</pre>

<hr style="border:none; border-top:1px solid #2c2c2c; margin:18px 0;">

<h2 id="citation" style="color:#4FC3F7;">Citation</h2>
<p style="color:#cfd8dc;">If you use this repository in academic or clinical research, please cite it.</p>

<pre style="background:#0b0d10; color:#e3f2fd; padding:14px; border-radius:10px; border:1px solid #23262e; overflow:auto;">
@misc{t1d_digital_twin_workshop,
  title        = {T1D Data-Based Digital Twin Workshop},
  author       = {Bustos, Oriol; Mujahid, Omer},
  year         = {2025},
  url          = {https://github.com/oriolbustos/t1d-data-based-digital-twin}
}
</pre>

<hr style="border:none; border-top:1px solid #2c2c2c; margin:18px 0;">

<div style="margin-top:16px; background:#0f1115; border:1px solid #23262e; border-radius:12px; padding:12px; display:flex; align-items:center; justify-content:space-between;">
  <span style="color:#90a4ae;">Maintained by MICELab • University of Girona</span>
  <a href="LICENSE.md" style="text-decoration:none; background:#1e88e5; color:#fff; padding:8px 12px; border-radius:8px;">View License</a>
</div>
