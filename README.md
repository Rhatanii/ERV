# Emotion-Coherent-Reasoning

Our ERV project handles interpretable and consistent explanation about Multimodal Emotion Recognition.

---

## 🚀 Environment Setup

Follow these steps to set up the environment required to run this project.

1.  **Base Repository Installation**
    First, follow the installation guide from the main [R1-V repository](https://github.com/StarsfieldAI/R1-V).

2.  **Install Flash Attention**
    ```bash
    pip install flash_attn==2.7.4.post1
    ```

3.  **Install Additional Python Libraries**
    ```bash
    pip install imageio decord moviepy==1.0.3 ipdb==0.13.13 h5py
    ```

---

##  checkpoint 📂 Model Checkpoints

Checkpoints will be uploaded soon.

---

## 🗂️ Folder Structure

[TODO: Add a description of your project's folder structure here.]

---

## ⚡ Inference

Run inference with ERV to generate interpretable and consistent emotional reasoning.

```bash
# Example inference command
bash eval_shard.sh

## Acknowledgement
This repository is built upon [R1-Omni](https://github.com/HumanMLLM/R1-Omni), [R1-V](https://github.com/StarsfieldAI/R1-V), and [HumanOmni](https://github.com/HumanMLLM/HumanOmni). We appreciate the open-source of the projects.
