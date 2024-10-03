# ViT Image Classification WebApp
 The Vision Transformer (ViT) model represents a revolutionary approach to image classification, leveraging the strengths of Transformer architectures that have been highly successful in NLP. Unlike traditional Convolutional Neural Networks (CNNs), ViT splits images into smaller patches and processes them like sequences, similar to words in text processing. This allows ViT to capture global context and intricate patterns across the entire image using self-attention mechanisms.

ViT operates by dividing an image into fixed-size patches (e.g., 16x16 pixels), flattening them, and then embedding these patches into a linear sequence that is fed into a transformer model. Unlike CNNs, which rely on local receptive fields and gradual spatial understanding through convolution layers, ViT has direct access to all parts of the image from the start. This enables it to model long-range dependencies between parts of the image more effectively. This architecture is especially efficient for large-scale datasets.

The ViT model has shown excellent performance on the ImageNet dataset, which contains over 1,000 object classes, making it well-suited for single-Image Classification tasks. ViT significantly reduces the need for extensive convolution layers, focusing instead on global self-attention mechanisms that are more flexible in identifying features across different regions of the image. This architecture allows ViT to handle complex classification tasks with fewer computational resources compared to traditional deep CNNs, while maintaining high accuracy.

## ViT Architecture Overview:

**Patch-Based Input Representation:** The ViT model starts by dividing an input image into fixed-size patches, which are then flattened into vectors. This unique approach allows the model to treat image patches as tokens, similar to how text is processed in natural language processing (NLP).

**Transformer Encoder Layers:** These flattened patches are processed through a series of Transformer encoder layers. Each layer consists of multi-head self-attention mechanisms and feed-forward neural networks. This architecture enables the model to learn complex relationships among patches and capture global contextual information across the image.

**Classification Head:** The output from the Transformer encoders is fed into a classification head that predicts the class of the object in the image. ViT's capacity to classify images into 1,000 distinct categories demonstrates its versatility and effectiveness in handling diverse visual data.

### Model Overview
- **Model Type:** Image classification using Vision Transformer (ViT).
- **Input Resolution:** 224x224 pixels for the base model.
- **Checkpoint:** Pretrained on ImageNet (1k or 21k datasets).
- **Number of Parameters:** 86.6 million (ViT-B/16 model).
- **Model Size:** 330 MB (ViT-B/16).
- **Patch Size:** 16x16 pixels per patch for standard configuration.
- **Pretraining:** Large-scale pretraining on ImageNet-21k for enhanced performance.
- **Self-Attention:** Global self-attention used instead of convolutions.
- **Performance:** Achieves state-of-the-art results on large datasets.
- **Model Variants:** ViT-B (Base), ViT-L (Large), and ViT-H (Huge), which differ in terms of patch size, hidden dimensions, and depth of the transformer layers.

## Prerequisites 
- `Python >= 3.8`
- `Ubuntu 20.04`
- `Anaconda`

### Installing Conda on WSL2.

To install Conda on WSL2, follow the link : [***here***](https://thesecmaster.com/step-by-step-guide-to-install-conda-on-ubuntu-linux/)

•	Once Conda is installed, you can create a new environment with the following command:

    $ conda create --name <env_name> python=3.8
•	To activate conda environment

    $ conda activate “env_name”
•	To view the list of available Conda environments, use:

    $ conda env list

## Usage
1. Clone the repository: 
   ```bash
   ubuntu@user:git clone https://github.com/krishnapriya-nynaru/ViT-Image-Classification-WebApp.git
2. Unzip the downloaded file: 
   ```bash
   ubuntu@user:unzip ViT-Image-Classification-WebApp.git
3. . Install the required packages: 
   ```bash
   ubuntu@user:pip install -r requirements.txt
4. Navigate to the project directory: 
   ```bash
   ubuntu@user:cd ViT_Image_Classification
5. Download the ViT TFLite model from the Qualcomm AI Hub and save it in the ViT_Image_Classification folder. You can access the download link here: [***ViT Model - Qualcomm AI Hub***](https://aihub.qualcomm.com/models/vit)
6. To run the image classification web app:
    ```bash
    ubuntu@user:python app.py
7. Open the Web App: Open the URL http://127.0.0.1:5000 in any web browser.
8. Click on "Choose File" and select the video file. Once the video is uploaded, click on "Run Inference."
9. The inference parameters and the output predicted class, along with confidence levels, will be displayed in the sidebar.

## Results
Below are some results of developed Application on test videos:-


![alt text](https://github.com/krishnapriya-nynaru/ViT-Image-Classification-WebApp/blob/main/ViT-Image-Classification/outputs/result1.png) 
![alt text](https://github.com/krishnapriya-nynaru/ViT-Image-Classification-WebApp/blob/main/ViT-Image-Classification/outputs/result2.png) 

### References
https://arxiv.org/pdf/2010.11929
