[Stock Movement Prediction Used Bi-typed Hybrid-relational Market Knowledge Graph via Dual Attention Networks]()



## 📌 Introduction
This project focuses on **predicting stock movements** by leveraging a **bi-typed hybrid-relational market knowledge graph** and **Dual Attention Networks (DAN)**. The approach integrates **graph-based modeling** and **deep learning techniques** to improve accuracy in financial market predictions.

## 🎯 Objectives
- Construct a **market knowledge graph** incorporating stock and executive relationships.
- Used **Momentum spillover effect** for better model.
- Utilize **Dual Attention Networks (DAN)** for feature extraction and relation learning.
- Enhance prediction accuracy using **graph-based deep learning methods**.
- Implement and evaluate the model in **PyTorch Geometric**.

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch Geometric
- **Graph Processing**: NetworkX
- **Machine Learning Libraries**: NumPy, Scikit-learn, Matplotlib
- **Data Handling**: Pickle, Pandas

## 🚀 Installation & Setup
### 1️⃣ Install Dependencies
```bash
pip install torch torchvision torchaudio torch-geometric numpy scikit-learn matplotlib networkx dill
```
OR

pip install -r requirements.txt


### 2️⃣ Download the Dataset
Ensure the **data** folder contains processed stock and executive relations.

The two datasets for SMP with their folder names are given below.

CSI100E
CSI300Epip install -r requirements.txt


### 3️⃣ Run the Training Script
```bash
python main1.py
```

## 🔍 Execution Workflow
### 📌 Data Processing
- Load **stock-executive interaction data** (`interactive.pkl`).
- Construct **market knowledge graphs** (explicit & implicit relations).
- Generate **graph-based features** for model training.

### 📌 Model Training
- Utilize **Dual Attention Networks (DAN)** for feature extraction.
- Train the model on **hybrid-relational graph data**.
- Optimize using **Adam optimizer** and **cross-entropy loss**.

### 📌 Model Evaluation
- Evaluate **AUC** (Area Under Curve) and **Accuracy** on test data.
- Example output from training:
  ```
  epoch32, train_loss=0.6495, eval_auc=0.5456, eval_acc=0.5189, test_auc=0.5516, test_acc=0.5274
  epoch50, train_loss=0.6014, eval_auc=0.5468, eval_acc=0.5356, test_auc=0.5325, test_acc=0.5233
  ```

## 📊 Results Visualization
- The model constructs **market knowledge graphs** showing **stocks & executive relations**.
- Performance metrics (AUC, accuracy) are plotted to assess training progress.
- Example **knowledge graph visualization**:
  ```python
  import networkx as nx
  import matplotlib.pyplot as plt
  G = nx.Graph()
  # Add stock & executive nodes, then visualize the graph
  nx.draw(G, with_labels=True, node_size=600, font_size=8)
  plt.show()
  ```

## 🔥 Challenges & Solutions
| Challenge                 | Solution |
|---------------------------|----------|
| Large-scale graph data    | Efficient data preprocessing & sparse tensor storage |
| Model overfitting         | Regularization (Dropout, BatchNorm) |
| API Latency in real-time  | Optimized graph sampling techniques |

## 📈 Future Improvements
- Implement **Transformer-based graph learning** for better feature extraction.
- Deploy as a **real-time prediction API**.
- Enhance knowledge graph with **macro-economic indicators**.

## 📖 References
1. PyTorch Geometric Docs - [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
2. NetworkX Graph Visualization - [https://networkx.org/](https://networkx.org/)
3. Dual Attention Networks Research - [https://arxiv.org/](https://arxiv.org/)

📌 **GitHub Repository** : *https://github.com/Ayussh-Raj/Stock-Movement-Prediction/tree/main*

