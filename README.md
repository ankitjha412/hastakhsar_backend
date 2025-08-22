# ✍️ Hastakhar - Handwriting & Signature Forgery Detection
This project is a **FastAPI backend** with a React frontend for verifying **signature & handwriting authenticity** using a **Siamese Neural Network** (PyTorch).

## 🚀 Features
- Upload **Reference** and **Test** images  
- Choose **task**: signature or handwriting  
- Uses **Siamese Network** trained in PyTorch  
- Returns **distance score** + **label** (Genuine/Forged)  
- Fully deployable on **Render** (free or paid tier)  
- Interactive API docs via **Swagger UI** (`/docs`)  

## 📂 Folder Structure
HANDWRITING_FORGERY_DETECTION/
│── data/ # training/testing datasets
│ ├── train/
│ └── test/
│
│── models/ # pre-trained PyTorch models
│ ├── siamese.pth
│ └── siamese_handwriting.pth
│
│── app.py # FastAPI entry point
│── model.py # Siamese network architecture
│── preprocess.py # Preprocessing utilities
│── verify.py # Verification helper
│── train.py # Model training script
│── requirements.txt # Python dependencies
│── .gitignore
│── venv/ # (local virtual environment - not pushed to git)

bash
Copy
Edit

## ⚙️ Installation (Local)
Clone the repository
```bash
git clone https://github.com/<your-username>/hastakhsar_backend.git
cd hastakhsar_backend
Create virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run FastAPI locally

bash
Copy
Edit
uvicorn app:app --reload
Now open:

API Docs → http://127.0.0.1:8000/docs

Root healthcheck → http://127.0.0.1:8000/

🌐 Deployment on Render
Push code to GitHub and make sure:

requirements.txt includes all dependencies

models/siamese.pth and models/siamese_handwriting.pth are committed (or uploaded separately)



