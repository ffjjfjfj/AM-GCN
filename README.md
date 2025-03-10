This project was submitted to the journal（）, and the official submission website is （）.
📖 Overview
AM-GCN（Enhanced Microexpression Recognition through Action Unit-Driven Graph Convolution and Stacked Attention Mechanisms） is a novel two-branch neural network designed for Micro-Expression Recognition (MER). It integrates a AA Module and a SA Module  to enhance spatial-temporal feature learning. The model achieves state-of-the-art performance on the CASME II dataset for both 3-class and 5-class micro-expression classification. 

Due to China Ministry of Science and Technologyunder  intellectual property protection, the training code will be uploaded after the paper is published。


🛠 Installation
Dependencies
Python 3.8+

PyTorch 1.10+

torchvision

pandas

OpenCV

numpy

bash

# Install required packages  
pip install torch torchvision pandas opencv-python numpy  
🚀 Usage
Dataset Preparation
Download CASME II Dataset: Request access from CASME II official site.

Directory Structure:


casme2/  
├── sub01/  
│   ├── video01/  
│   │   ├── img1.jpg  
│   │   ├── img2.jpg  
│   │   └── ...  
├── sub02/  
└── data.xlsx  # Annotation file  
Update Paths: Modify raf_path in CASME2_3.py and CASME2_5.py to your dataset path.

Training
3-class MER:

bash

python CASME2_3.py --batch_size 24 --lr 0.0001 --epochs 100  
5-class MER:

bash
python CASME2_5.py --batch_size 32 --lr 0.0008 --epochs 100  
Key Arguments
--raf_path: Path to CASME II dataset.

--batch_size: Batch size (default: 24/32).

--lr: Learning rate (default: 0.0001).

--epochs: Training epochs (default: 100).

🧩 Model Architecture

Input: Resized facial images (3x14x14).

Process: Vision Transformer encodes positional embeddings.

Output: 512x14x14 feature maps.

Layers: 90+ convolutional blocks with kernel sizes 1x1, 3x3, and stride variations.

Purpose: Extracts multi-scale motion features from apex-frame differences.

Encoder Block
Detailed structure of the encoder block.

