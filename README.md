# EngageNet
Dataset Link: Please contact abhinav@iitrpr.ac.in or monisha.21csz0023@iitrpr.ac.in to get access to EngageNet dataset. 

About the dataset:
- Large scale dataset EngageNet has around 31 hours of data of 127 subjectst (83 males and 44 females), and over 11.3K 10-sec clips.
- The participants were between the age range of 18 to 37 years.
- A web based platform was designed for the data collection study.
- The participant could use the web based interface anywhere they felt comfortable. This enabled collection of data in diverse environments.
- The participants in the videos were assigned to different engagement classes by a group of three annotators.
- The four different engagement classes are "Highly-Engaged", "Engaged", "Barely-Engaged", and "Not-Engaged".
- We created subject independent data split leading to 90 participants in Train, 11 participants in Validation, and 26 participants in Test.
- There are a total of 7983 Train, 1071 Validation, and 2257 Test videos.

Code, Models: Will be made available soon.

## Setup:

# Clone MARLIN somewhere local
git clone https://github.com/microsoft/MARLIN.git
cd MARLIN

# Install MARLIN dependencies
pip install -r requirements.txt

# (Optional) Install FaceXZoo if not included
git clone https://github.com/JDAI-CV/FaceX-Zoo.git FaceXZoo

## 🚀 Quick Start

# Clone the repo
git clone https://github.com/<yourusername>/engagenet_baselines.git
cd engagenet_baselines

# Create environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# (Optional) setup MARLIN + FaceXZoo
mkdir -p .marlin && cd .marlin
git clone https://github.com/microsoft/MARLIN.git
cd MARLIN && git clone https://github.com/JDAI-CV/FaceX-Zoo.git FaceXZoo

# Run live demo
python live_marlin_openface_combination.py
