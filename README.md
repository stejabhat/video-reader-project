# 🎥 Video Reader using FastVLM

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Python project that leverages **FastVLM** to analyze and interpret video content efficiently. Perfect for video understanding, research, or building AI-powered media tools.

---

## 🚀 Features

- Load and analyze videos frame by frame  
- Extract insights using FastVLM’s pre-trained models  
- Easy-to-use Python interface (`analyze_video.py`)  
- Lightweight and modular — integrate into your projects easily  

---

## 💻 Setup

1. **Clone FastVLM repository**:

```bash
git clone https://github.com/<fastvlm-repo>
cd fastvlm
bash get_models.sh
```

## Clone this Video Reader project:

git clone https://github.com/stejabhat/video-reader-project.git
cd video-reader-project

## Install dependencies (optional, if using virtual environment):
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# how to run
python analyze_video.py --video <path_to_your_video>
