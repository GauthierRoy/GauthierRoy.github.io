---
title: "Deep Learning Mastery: From Foundations to Advanced Generative Models with PyTorch"
excerpt: "Implemented, trained, and evaluated diverse **deep learning** models (**MLPs**, **CNNs**, **Transformers**, **GANs**, **Diffusion Models**) using **PyTorch** and **NumPy** for tasks like **image classification/generation**, **sequence modeling**, and **robotic control**."
collection: portfolio
header:
  teaser: "/images/pytorch-journey/gan_final_svhn.png" # Using Figure 12 from the report
# permalink:  /portfolio/pytorch-deep-learning-journey/
---

This project represents a comprehensive journey through modern deep learning, systematically building understanding and practical skills from the ground up. Starting with fundamental neural network mechanics implemented purely in NumPy, the work progressed to building, training, and analyzing sophisticated models like Transformers, GANs, and Diffusion Models using PyTorch on diverse datasets and tasks.

### 1. Building Blocks: Neural Networks from Scratch (NumPy)

*   Solidified understanding of core deep learning mechanics by implementing essential components—activation functions (ReLU, Sigmoid), loss functions (Softmax, Cross-Entropy), optimizers (SGD with L2 regularization), and the backpropagation algorithm—using only NumPy.
*   Applied these foundational elements to train simple models (Softmax Regression, 2-Layer MLP) on the MNIST dataset, developing crucial utilities for data handling, batching, and training visualization (loss/accuracy curves). This stage built a strong intuition for gradient-based learning.

### 2. Image Recognition: Convolutional Neural Networks (CNNs)

*   Explored CNNs by first implementing key modules (2D Convolution, Max Pooling) from scratch, deepening the understanding of spatial feature extraction and gradient flow in convolutional layers.
*   Transitioned to PyTorch, leveraging its `nn.Module` system to construct and train standard and custom CNN architectures on the CIFAR-10 image classification task. This involved implementing robust training and validation loops and managing hyperparameters effectively. Introduced SGD with momentum for improved optimization.

### 3. Sequence Understanding: Transformers

*   Delved into sequence modeling by implementing the Transformer architecture using PyTorch. Developed core components including multi-head self-attention (GPT-2 style), positional encodings, and encoder/decoder blocks with residual connections and layer normalization.
*   Implemented distinct attention masking strategies for Encoder (bi-directional for full context) and Decoder (causal/unidirectional for auto-regressive generation) models.
*   Successfully built and trained Encoder-only, Decoder-only, and combined Encoder-Decoder models on tasks like numerical sorting and machine translation (English-to-Pig-Latin), demonstrating proficiency in handling sequential data and attention mechanisms. Implemented efficient auto-regressive generation using temperature scaling and caching.

### 4. Creative Generation: GANs & Diffusion Models

*   **Generative Adversarial Networks (GANs):** Implemented a GAN for generating images on the SVHN dataset.
    *   Designed and trained competing Generator (CNN-based upsampler using Transposed Convolutions) and Discriminator networks.
    *   Implemented the adversarial training loop, directly applying the **minimax loss formulation from the original Goodfellow et al. GAN paper** (using the non-saturating generator objective) to drive the generator towards producing realistic images.
    *   Achieved successful generation of SVHN-like digits, showcasing the ability to implement and stabilize GAN training.
    *   ![Final GAN Output](/images/pytorch-journey/gan_final_svhn.png)

*   **Denoising Diffusion Probabilistic Models (DDPMs):** Implemented and applied cutting-edge diffusion models to image generation and robotics.
    *   Developed a `NoiseScheduler` module implementing both the forward (noise addition) and reverse (denoising) processes, **directly translating the core mathematical framework (Equations & Algorithms) presented in the original Ho et al. DDPM paper**.
    *   Implemented the diffusion model training, optimizing a U-Net to predict noise based on the **simplified variational lower bound objective (Eq. 14 / Algorithm 1 from the DDPM paper)**.
    *   Mastered **Classifier-Free Guidance (CFG)**, implementing the modified sampling process (**following Eq. 6 in the CFG paper**) to enhance sample quality and class-conditionality for CIFAR-10 image generation.
    *   Implemented **dynamic thresholding** during sampling (predicting `x_0` via **DDPM Eq. 15**, clamping, then deriving `x_{t-1}` via **DDPM Eq. 6/7**, as explored in improved DDPM literature) to prevent color saturation artifacts common with high guidance weights.
    *   ![Diffusion Image Output](/images/pytorch-journey/diffusion_cifar_g2_thresh.png)
    *   Extended diffusion models beyond images, applying them to **robotic control** on the Push-T trajectory dataset. Implemented action chunking and successfully trained a policy capable of generating smooth, effective, multi-step actions to solve the task, demonstrating the versatility of diffusion models.

### Key Skills & Technologies Demonstrated:

*   **Deep Learning Fundamentals:** Strong grasp of backpropagation, gradient descent, loss functions, activation functions, regularization.
*   **Model Implementation:** Ability to implement diverse architectures (MLP, CNN, Transformer, GAN, Diffusion) from scratch and using PyTorch.
*   **PyTorch Proficiency:** Effective use of `torch.nn`, `torch.optim`, `DataLoader`, automatic differentiation, GPU acceleration.
*   **Training & Evaluation:** Expertise in designing training loops, implementing validation strategies, hyperparameter tuning, and visualizing results.
*   **Generative Modeling:** Hands-on experience with GANs and Diffusion Models, including implementing core algorithms based on seminal research papers and advanced techniques like CFG and thresholding.
*   **Applications:** Applied deep learning to image classification (MNIST, CIFAR-10, SVHN), sequence processing (sorting, translation), image generation, and robotics.
*   **Libraries:** PyTorch, NumPy, Matplotlib, Hugging Face libraries (`datasets`, `tokenizers`), Scikit-learn (concepts).