# 🧠 Research Topic Classification using GCN

An end-to-end deep learning project that leverages **Graph Convolutional Networks (GCNs)** to classify research papers into different topics using the **Cora citation network**. The project integrates graph-based learning, PyTorch, ONNX model export, and efficient inference through ONNX Runtime and FastAPI.

🔗 **GitHub Repository:** [CORA-PROJECT](https://github.com/snape0931/CORA-PROJECT)

## ✨ Features

- 🕸️ **Graph-Based Learning:** Learns from research paper features and citation relationships.
- 🧠 **GCN Model:** Implements Graph Convolutional Networks using PyTorch.
- 📚 **Cora Dataset:** Classifies research papers into seven topic categories.
- ⚡ **ONNX Export:** Converts the trained model into a portable ONNX format.
- 🚀 **ONNX Runtime:** Supports efficient model inference.
- 🌐 **FastAPI Backend:** Provides an API for generating predictions.
- 💻 **Web Interface:** Offers an interface to interact with the model.
- 🔬 **Jupyter Notebook:** Supports model development and experimentation.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | GCN model development and training |
| Cora Dataset | Citation network for classification |
| ONNX | Model export and interoperability |
| ONNX Runtime | Model inference |
| FastAPI | Backend API |
| Jupyter Notebook | Model development and experimentation |
| HTML, CSS, JavaScript | Web interface |

## 📊 Dataset

The project uses the **Cora citation network**, a benchmark dataset for graph-based machine learning.

- 🔵 **Nodes:** Research papers
- 🔗 **Edges:** Citation relationships between papers
- 🧩 **Node Features:** Numerical representations of research papers
- 🎯 **Objective:** Predict the research topic of a paper using its features and citation network.

## 🏷️ Classification Categories

The model classifies research papers into seven categories:

1. 📖 Case-Based
2. 🧬 Genetic Algorithms
3. 🧠 Neural Networks
4. 📈 Probabilistic Methods
5. 🎮 Reinforcement Learning
6. 🌳 Rule Learning
7. 📐 Theory

## 🔄 Project Workflow

1. 📥 Load and preprocess the Cora citation dataset.
2. 🕸️ Represent research papers and their citation relationships as a graph.
3. 🧠 Train the Graph Convolutional Network using PyTorch.
4. 📦 Export the trained model to ONNX format.
5. ⚡ Load the exported model using ONNX Runtime.
6. 🎯 Generate research topic predictions.
7. 🌐 Integrate model inference with the FastAPI backend and web interface.
8. ☁️ Deploy the application for online access.

## 📂 Project Structure

```text
CORA-PROJECT/
├── static/                     # Frontend assets
├── anaconda_projects/db/       # Project data directory
├── .ipynb_checkpoints/         # Jupyter notebook checkpoints
├── gcn_1.ipynb                 # Model development notebook
├── main.py                     # FastAPI application
├── cora_data.npz               # Dataset file
├── simple_gcn_cora.onnx        # Exported GCN model
├── simple_gcn_cora.onnx.data   # ONNX model data
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🚀 Deployment

The application was deployed on **Render**. However, performance was significantly affected by the resource and bandwidth limitations of Render's free tier.

You can run the project locally to access the web interface and test the API without relying on the public deployment.

## 💻 Run Locally

Follow these steps to run the project on your machine.

### 📋 Prerequisites

- Python 3.10 or a compatible version
- Git
- pip

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/snape0931/CORA-PROJECT.git
cd CORA-PROJECT
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Start the FastAPI Server

```bash
python -m uvicorn main:app --reload
```

### 5️⃣ Access the Application

Once the server starts, open the following URLs in your browser:

- 🌐 **Web Interface:** http://127.0.0.1:8000
- 📖 **Interactive API Documentation:** http://127.0.0.1:8000/docs

The web interface is served locally by the FastAPI application. Keep the terminal running while using the application.
**Note:** Local execution does not require a public deployment. The application is accessible through the local URLs while the FastAPI server is running.

## 👩‍💻 Author

**Saumya Srivastava**
