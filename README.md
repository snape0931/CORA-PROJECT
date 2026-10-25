# 🧠 Research Topic Classification using GCN

An end-to-end deep learning project that leverages **Graph Convolutional Networks (GCNs)** to classify research papers into different topics using the **Cora citation network**. This project combines graph-based learning, PyTorch, ONNX model export, and efficient inference.

## 🚀 Deployment Note

⚠️ The application was deployed on Render. However, performance was severely affected by the bandwidth and resource limitations of Render's free tier.
- 💻 **GitHub Repository:** [CORA-PROJECT](https://github.com/snape0931/CORA-PROJECT)

## ✨ Features

- 🕸️ **Graph-Based Learning:** Learns from paper features and citation relationships.
- 🧠 **GCN Model:** Implements Graph Convolutional Networks using PyTorch.
- 📚 **Cora Dataset:** Classifies research papers into seven topic categories.
- ⚡ **ONNX Export:** Exports the trained model for portable inference.
- 🚀 **ONNX Runtime:** Supports efficient model inference.
- 🌐 **FastAPI Backend:** Provides an API for model predictions.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | GCN model development |
| Cora Dataset | Citation network for classification |
| ONNX | Model export and interoperability |
| ONNX Runtime | Model inference |
| FastAPI | Backend API |
| Jupyter Notebook | Model development and experimentation |

## 📂 Project Structure

```text
CORA-PROJECT/
├── 📁 static/
├── 📓 gcn_1.ipynb
├── 📓 Untitled.ipynb
├── 🐍 main.py
├── ⚡ simple_gcn_cora.onnx
├── 📦 simple_gcn_cora.onnx.data
└── 📋 requirements.txt
```

## 📊 Dataset

The project uses the **Cora citation network**, a benchmark dataset for graph-based machine learning.

- 🔵 **Nodes:** Research papers
- 🔗 **Edges:** Citation relationships between papers
- 🧩 **Node Features:** Numerical representations of papers
- 🎯 **Objective:** Predict the research topic of a paper

### 🏷️ Classification Categories

1. 📖 Case-Based
2. 🧬 Genetic Algorithms
3. 🧠 Neural Networks
4. 📈 Probabilistic Methods
5. 🎮 Reinforcement Learning
6. 🌳 Rule Learning
7. 📐 Theory

## 🔄 Project Workflow

1. 📥 Load and preprocess the Cora citation dataset.
2. 🕸️ Represent research papers and citations as a graph.
3. 🧠 Train the GCN model using PyTorch.
4. 📦 Export the trained model to ONNX format.
5. ⚡ Load the exported model using ONNX Runtime.
6. 🎯 Generate research topic predictions.
7. 🌐 Integrate model inference with the application interface.
8. ☁️ Deploy the backend and web application.

## 👩‍💻 Author

**Saumya Srivastava**

🔗 [GitHub Profile](https://github.com/snape0931)

