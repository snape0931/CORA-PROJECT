🧠 Research Topic Classification using GCN

An end-to-end deep learning project that leverages Graph Convolutional Networks (GCNs) to classify research papers into different topics using the Cora citation network. This project combines graph-based learning, PyTorch, ONNX model export, and efficient inference.

🚀 Deployment Note

⚠️ The application was deployed on Render. However, performance was severely affected by the bandwidth and resource limitations of Render’s free tier.

You can run the project locally by following the setup instructions below.

* GitHub Repository: CORA-PROJECT

✨ Features

* 🕸️ Graph-Based Learning: Learns from paper features and citation relationships.
* 🧠 GCN Model: Implements Graph Convolutional Networks using PyTorch.
* 📚 Cora Dataset: Classifies research papers into seven topic categories.
* ⚡ ONNX Export: Exports the trained model for portable inference.
* 🚀 ONNX Runtime: Supports efficient model inference.
* 🌐 FastAPI Backend: Provides an API for model predictions.
* 💻 Local Execution: Run the application and test API endpoints on your own machine.

🛠️ Tech Stack

Technology	Purpose
Python	Core programming language
PyTorch	GCN model development
Cora Dataset	Citation network for classification
ONNX	Model export and interoperability
ONNX Runtime	Model inference
FastAPI	Backend API
Jupyter Notebook	Model development and experimentation

📂 Project Structure

CORA-PROJECT/
├── static/                     # Frontend assets
├── anaconda_projects/db/       # Project data directory
├── .ipynb_checkpoints/         # Jupyter checkpoints
├── gcn_1.ipynb                 # Model development notebook
├── main.py                     # FastAPI application
├── cora_data.npz               # Dataset file
├── simple_gcn_cora.onnx        # Exported GCN model
├── simple_gcn_cora.onnx.data   # ONNX model data
├── requirements.txt            # Python dependencies
└── README.md

📊 Dataset

The project uses the Cora citation network, a benchmark dataset for graph-based machine learning.

* 🔵 Nodes: Research papers
* 🔗 Edges: Citation relationships between papers
* 🧩 Node Features: Numerical representations of papers
* 🎯 Objective: Predict the research topic of a paper

🏷️ Classification Categories

1. 📖 Case-Based
2. 🧬 Genetic Algorithms
3. 🧠 Neural Networks
4. 📈 Probabilistic Methods
5. 🎮 Reinforcement Learning
6. 🌳 Rule Learning
7. 📐 Theory

🔄 Project Workflow

1. 📥 Load and preprocess the Cora citation dataset.
2. 🕸️ Represent research papers and citations as a graph.
3. 🧠 Train the GCN model using PyTorch.
4. 📦 Export the trained model to ONNX format.
5. ⚡ Load the exported model using ONNX Runtime.
6. 🎯 Generate research topic predictions.
7. 🌐 Integrate model inference with the application interface.
8. ☁️ Deploy the backend and web application.

💻 Run Locally

Follow these steps to run CORA-PROJECT on your local machine.

Prerequisites

* Python 3.10 or a compatible version
* Git
* pip

1. Clone the Repository

git clone https://github.com/snape0931/CORA-PROJECT.git
cd CORA-PROJECT

2. Create a Virtual Environment

python -m venv venv

Windows

venv\Scripts\activate

macOS/Linux

source venv/bin/activate

3. Install Dependencies

python -m pip install --upgrade pip
pip install -r requirements.txt

4. Start the FastAPI Server

python -m uvicorn main:app --reload

5. Access the Application

Once the server starts, open the following URLs in your browser:

* Web Interface: http://127.0.0.1:8000
* Interactive API Documentation: http://127.0.0.1:8000/docs

The web interface is served locally by the FastAPI application. Keep the terminal running while using the application.

Troubleshooting

* Ensure all dependencies are installed successfully.
* Keep simple_gcn_cora.onnx and simple_gcn_cora.onnx.data together in the expected location.
* Make sure any required dataset files are available.
* If the server fails to start, check the error message and verify the Python package versions.

Note: Local execution does not require a public deployment. The application is accessible through the local URLs while the server is running.

👩‍💻 Author

Saumya Srivastava

🔗 GitHub Profile
