# The-Long-Range-Depth-Estimation

This project is a derivative work based on [Monocular-Depth-Estimation-Toolbox](https://github.com/Ma-Zhuang/DepthFormer), 
which is licensed under the Apache License 2.0.

## 📖 Introduction

**Monocular depth estimation**, a cost-effective solution for reconstructing 3D structures from single 2D images, finds widespread applications in robotic navigation, UAV control, and intelligent manufacturing.

Despite significant advancements in deep neural networks, accurately estimating depth for **distant objects** remains challenging due to insufficient global context modeling.

To address this, we introduce a novel monocular depth estimation framework featuring:

* **CNN-Transformer dual-branch encoder** for local-global feature extraction.
* A **Cross-Dimensional Feature Fusion (CSF) Module** for enhanced global depth understanding.
* In the decoder, a **Depthwise Separable Upsampling Block (DSUB)** and **Multi-Scale Self-Attention Module (MSA)** to refine upsampling and recover detailed spatial information.

**Experimental results** demonstrate superior long-range prediction accuracy on KITTI and SUN RGB-D datasets.

---
## 📊 Method Overview

### 📌 Architecture Overview

📷 [![总体架构1](https://github.com/user-attachments/assets/724e8152-0d8c-4ba0-ae4b-fd15c8a013d9)

]

Illustration: Dual-branch Encoder (CNN + Transformer), CSF Fusion Module, DSUB & MSA-based Decoder

---

## 📊 Benchmark Performance

| Dataset       | Abs-Rel 🔽 | RMSE 🔽   | Notes                                                      |
| :------------ | :--------- | :-------- | :--------------------------------------------------------- |
| **KITTI**     | **0.050**  | **2.111** |                                                            |
| **NYU-Depth** | **0.094**  | **0.329** |                                                            |
| **SUN RGB-D** | **0.142**  | **0.433** | Using NYU pre-trained weights, shows strong generalization |

---

## 📊 Qualitative Results

 📷 **\[![image](https://github.com/user-attachments/assets/8d2871ea-8258-4056-9742-5a4b11074995)
]**

---

## 📦 Installation

### 📋 Prerequisites

* **Linux** or **macOS** (Windows experimental support)
* Python **3.6+**
* PyTorch **1.3+**
* CUDA **9.2+** (or CUDA 9.0 if building from source)
* GCC **5+**
* MMCV

### 🛠️ Installation Steps

Tested with:

* PyTorch 1.8.0
* CUDA 11.1
* Python 3.8
* Ubuntu 20.04

If your environment is similar, follow these steps:

```bash
conda create -n MDE python=3.8
conda activate MDE

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/sarielbit/The-Long-Range-Depth-Estimation.git
cd The-Long-Range-Depth-Estimation
pip install -e .

pip install future tensorboard
```

---

## 📂 Dataset Preparation
### 🔹 KITTI

From the official website, download the KITTI dataset. Then unzip the files into `data/kitti`. Finally, copy split files (whose names start with `kitti`) in the `splits` folder into `data/kitti`.

### 🔹 NYU Depth v2

Download from: [Google Drive Link](https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing)

Then, download the standard test set from the official site:
[NYU Depth v2 Dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

> ⚠️ The downloaded file will unzip into `train` and `test` folders.

### 🔹 SUN RGB-D

Download from: [SUNRGBD Official Download](https://rgbd.cs.princeton.edu/)

After downloading, place the dataset under `data/SUNRGBD/`.

### 🔹 Example Folder Structure:

```
The-Long-Range-Depth-Estimation
├── depth
├── tools
├── configs
├── splits
├── data
│   ├── kitti
│   │   ├── input
│   │   │   ├── 2011_09_26
│   │   │   ├── ...
│   │   ├── gt_depth
│   │   │   ├── 2011_09_26_drive_0001_sync
│   │   │   ├── ...
│   │   ├── split_file.txt
│   ├── nyu
│   │   ├── basement_0001a
│   │   ├── ...
│   │   ├── split_file.txt
│   ├── SUNRGBD
│   │   ├── SUNRGBD
│   │   │   ├── kv1
│   │   │   ├── kv2
│   │   │   ├── ...
│   │   ├── split_file.txt
```

## 🚀 Usage

### 🏋️ Training

**Single / Multi-GPU Training:**

```bash
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

**Optional Arguments:**

* `--no-validate`
* `--work-dir ${WORK_DIR}`
* `--resume-from ${CHECKPOINT_FILE}`
* `--load-from ${CHECKPOINT_FILE}`
* `--deterministic`

**Example:**

```bash
bash ./tools/dist_train.sh configs/long_rangedepth/swinl_w7_22k_kitti.py 3 --work-dir work_dirs/saves/long_rangedepth/swinl_w7_22k_kitti
```

### 🦪 Inference & Testing

**Single-GPU Testing:**

```bash
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show]
```

**Example:**

```bash
python tools/test.py configs/long_rangedepth/swinl_22k_w7_kitti.py \
checkpoints/swinl_22k_w7_kitti.pth \
--show-dir swinl_22k_w7_kitti_results
```

---


## 📖 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{thelongrangedepth2025,
  title={Enhancing Long-Range Depth Estimation via Dual-Branch CNN-Transformer Integration and Multi-Dimensional Feature Fusion},
  author={Yunhao Chen and Qian Yin},
  journal={The Visual Computer},
  year={2025},
  note={Submitted}
}
```

---

## 📩 Contact

For questions and issues, feel free to open an [Issue](https://github.com/your-repo/The-Long-Range-Depth-Estimation/issues) or reach out via email: `your_email@example.com`

---

