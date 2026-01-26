# ![Logo](banner.svg "X-Min Learn")

**X-Min Learn** is an open source, standalone software designed for the automatic identification of mineral phases from X-ray map data. It offers a comprehensive and interactive environment for analyzing your data using **machine learning** classifiers and **image analysis** algorithms.

The software also includes tools for **developing** custom machine learning models with a fully **no-code** approach. You can:
* Generate _ground truth datasets_ from your selected rock samples
* Train _models_ tailored to your specific research needs
* Evaluate their _performance_ using statistical metrics and interactive graphics
* Automate future _classifications_ through your personally trained classifiers

## Installation

| Platform                 | Method                                                              | GPU acceleration                                   |
| ------------------------ | ------------------------------------------------------------------- | -------------------------------------------------- |
| Windows (CUDA 12.8+ GPU) | [Installer](#option-1--windows-installer-gpu-ready) (recommended)   | ✅ Built-in                                       |
| Windows (older/no GPU)   | [From source](#option-2--install-from-source-windows--macos--linux) | Optional (choose PyTorch variant)                  |
| macOS                    | [From source](#option-2--install-from-source-windows--macos--linux) | ❌ CPU only (MPS support currently not available) |
| Linux                    | [From source](#option-2--install-from-source-windows--macos--linux) | Optional (choose PyTorch variant)                  |

CUDA compatibility depends on your GPU model and the installed NVIDIA drivers. [Here](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) you can check the minimum required driver versions. Check your current driver version from the NVIDIA Control Panel.

> [!NOTE]
> Installing X-Min Learn with GPU acceleration support will increase the speed of model training. CUDA dlls require ~4 GB extra space on your disk.

### Option 1 – Windows Installer (GPU-ready)

1. Download Windows Installer for the [latest](https://github.com/albdag/X-Min-Learn/releases/latest) X-Min Learn version.
2. Run the installer and follow the setup wizard.
3. The installer deploys PyTorch GPU runtime DLLs targeting  **CUDA 12.8+** .

> [!WARNING]
> Your NVIDIA GPU and driver must support CUDA 12.8+. If you have an older GPU, no GPU, or prefer CPU-only mode, use [Option 2](#option-2--install-from-source-windows--macos--linux).

### Option 2 – Install from source (Windows / macOS / Linux)

> [!IMPORTANT]
> **Prerequisites:**
> * Python v.3.12.x
> * Git (optional)

#### Steps

1. **Download or clone the repository:**

   * _Download_:<br/>
   [https://github.com/albdag/X-Min-Learn/archive/refs/heads/main.zip](https://github.com/albdag/X-Min-Learn/archive/refs/heads/main.zip)<br/>
     
   * _Clone_:
     ```
     git clone https://github.com/albdag/X-Min-Learn.git
     ```
2. **Create and activate a virtual environment:**

   ```
   # Windows
   python -m venv path\to\new\virtual\environment
   path\to\new\virtual\environment\Scripts\activate

   # macOS / Linux
   python3 -m venv /path/to/new/virtual/environment
   source /path/to/new/virtual/environment/bin/activate
   ```
3. **Install base dependencies:**

   ```
   pip install -r path\to\repository\requirements.txt
   ```
4. **Install PyTorch (choose one):**

   | Scenario                 | Command                                                                         |
   | ------------------------ | ------------------------------------------------------------------------------- |
   | CPU only (all platforms) | `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu`   |
   | NVIDIA GPU, CUDA 12.8    | `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128` |
   | NVIDIA GPU, CUDA 11.8    | `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118` |

> [!NOTE]
> See [PyTorch – Previous Versions](https://pytorch.org/get-started/previous-versions/) for additional CUDA/PyTorch combinations. Be aware that _torch_ versions <2.6.0 have not been tested.
   
5. **Run the application:**

   ```
   cd path\to\repository
   python -m .\src\main.py
   ```

## Credits
Many of X-Min Learn icons are provided by [Icons8](https://icons8.it/).
