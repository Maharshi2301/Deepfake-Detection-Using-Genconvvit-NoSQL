# Deepfake-Detection-Using-Genconvvit-NoSQL
Maharshi Vyas, Devarsh Bharatiya and Harsh Patel

This repository contains the implementation code for **Deepfake Video Detection Using Generative Convolutional Vision Transformer (GenConViT)** paper. Find the full paper on arXiv [here](https://arxiv.org/abs/2307.07036).

<p style="text-align: justify;">
Deepfake technology has rapidly evolved, raising serious concerns about misinformation, cybersecurity, and media integrity [2]. Existing detection methods primarily rely on image-based analysis, which often fails to capture temporal inconsistencies in manipulated videos [1]. This project proposes an advanced deepfake detection system that integrates GenConViT, a hybrid Convolutional Vision Transformer (ConViT), with a NoSQL database to enhance accuracy, efficiency, and scalability. The system utilizes video segmentation, frame extraction, and deep feature analysis to detect manipulations more effectively. Unlike traditional approaches, it simultaneously analyzes spatial and temporal inconsistencies, ensuring precise classification [4]. Furthermore, the integration of NoSQL databases enables efficient storage and retrieval of large-scale video datasets, overcoming the limitations of SQL-based systems. Experimental results demonstrate high detection accuracy, improved processing speed, and scalability, making the system suitable for forensic investigations, media verification, and cybersecurity applications [10]. This research not only enhances deepfake detection capabilities but also establishes a scalable and adaptive foundation for future advancements in AI-driven video forensics.
</p>

## GenConViT Model Architecture
For the Detection Model we have used the GenConViT model. To use this model, review the provided research paper and its GitHub repository. https://github.com/erprogs/GenConViT 
The GenConViT model consists of two independent networks and incorporates the following modules:
<pre>
    Autoencoder (ed),
    Variational Autoencoder (vae), and
    ConvNeXt-Swin Hybrid layer
</pre>

The code in this repository enables training and testing of the GenConViT model for deepfake detection.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
  - [Model Training](#model-training)
  - [Model Testing](#model-testing)
- [Results](#results)

## Requirements
<pre>
    * Python 3.x
    * PyTorch
    * numpy
    * torch
    * torchvision
    * tqdm
    * decord
    * dlib
    * opencv
    * face_recognition
    * timm
</pre>

## Usage
<p>python -m venv venv</p>
<p>venv\Scripts\activate</p>
<p>pip install flask pymongo</p>
