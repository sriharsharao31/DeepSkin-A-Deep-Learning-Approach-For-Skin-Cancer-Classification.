# DeepSkin-A-Deep-Learning-Approach-For-Skin-Cancer-Classification.
📌 Project Overview
DeepSkin is an AI-powered deep learning project focused on detecting and classifying skin cancer from dermoscopic images using CNN architectures. It utilizes transfer learning with DenseNet169 and ResNet50 on the HAM10000 dataset to improve diagnostic accuracy and aid in early detection of skin cancer.

🧠 Motivation
Skin cancer is one of the most rapidly spreading cancers globally. Traditional diagnosis is time-consuming, expensive, and heavily reliant on expert judgment. This project aims to automate the detection process using deep learning, improving accuracy and enabling early interventions — especially in underserved areas.

🧪 Technologies & Tools
Programming Language: Python 3.x

Frameworks & Libraries: TensorFlow, Keras, PyTorch, OpenCV, Scikit-learn

Development Environment: Jupyter Notebook, Google Colab, PyCharm

Deployment Tools: Flask, Docker, Streamlit

Dataset: HAM10000 Dataset

🏗️ System Architecture
Key Features:
CNN-based image classification

Transfer learning with DenseNet169 & ResNet50

Image preprocessing: Sampling, dull razor, segmentation using autoencoder-decoder

Undersampling vs Oversampling comparison

Integration-ready for telemedicine platforms

Modules:
📁 Data Acquisition & Augmentation

🔍 Preprocessing (Segmentation, Noise Reduction)

🧠 Model Training (Transfer Learning)

📊 Evaluation (Accuracy, Precision, Recall, F1-Score)

🖥️ Web Interface / App Integration

📊 Results
Model	Sampling Method	Accuracy	F1-Score
DenseNet169	Undersampling	High	Good
ResNet50	Oversampling	Very High	High
The ResNet50 model with oversampling outperformed other models, offering high reliability in real-world applications.
🚀 Installation & Setup
Clone the repository:

bash
git clone https://github.com/your-username/DeepSkin.git
cd DeepSkin
Create a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies:

bash
pip install -r requirements.txt
Run the app (example using Streamlit):

bash
streamlit run app.py
📁 Dataset
Name: HAM10000

Samples: 10,015 dermoscopic images

Classes: 7 lesion types (melanoma, benign keratosis, etc.)

Preprocessing Includes:

Image resizing

Hair removal

Segmentation via autoencoders

🎯 Objectives
Automate classification of skin cancer from dermoscopic images

Improve diagnostic speed and reduce human error

Enable mobile and web-based access for early detection in remote areas

⚠️ Limitations
Bias in datasets (lighter skin tones are overrepresented)

Black-box nature of CNNs (lack of explainability)

Needs high-quality images for accurate predictions

Clinical deployment requires regulatory approvals

🔮 Future Scope
Integrate multimodal inputs (e.g., genetic data, history)

Implement federated learning for data privacy

Real-time inference on wearable/mobile devices

Expand model to detect other dermatological conditions

🤝 Team
R. Sri Harsha Rao (2111CS020431)

S. Sai Kiran Reddy (2111CS020432)

A. Sai Kiran (2111CS020433)

K. Sai Krishna Reddy (2111CS020434)

M. Charan Kumar (2111CS020435)

Mentor: Dr. Sujit Das, Assistant Professor, Malla Reddy University

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

