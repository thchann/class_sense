## Setup:

# Install MARLIN dependencies
- pip install -r requirements.txt

# (Optional) Install FaceXZoo if not included
- git clone https://github.com/JDAI-CV/FaceX-Zoo.git FaceXZoo

## 🚀 Quick Start

# Clone the repo
- git clone https://github.com/<yourusername>/engagenet_baselines.git
- cd engagenet_baselines

# Create environment
- python3 -m venv venv && source venv/bin/activate
- pip install -r requirements.txt

# (Optional) setup MARLIN + FaceXZoo
- mkdir -p .marlin && cd .marlin
- git clone https://github.com/microsoft/MARLIN.git
- cd MARLIN && git clone https://github.com/JDAI-CV/FaceX-Zoo.git FaceXZoo

# Run live demo
- python live_marlin_openface_combination.py
- Link to posted live demo: 
https://drive.google.com/drive/folders/1zaw-QntnvuwWmHRhLo98DzCJSy2f4hSe?dmr=1&ec=wgc-drive-hero-goto


## 📊 Results

We trained **EngageNet** on the DAiSEE dataset and achieved:

- **Validation Accuracy**: 80% stored (77.51% in random test run)

<p align="center">
  <img src="images/epoch200-0.7551acc.png" alt="Validation Accuracy Curve" width="500"/>
</p>