





# import streamlit as st
# import torch
# from train import SiameseNetwork
# from verify import verify_pair

# st.title("✍️ Handwriting & Signature Forgery Detector")

# task = st.radio("Select Task", ["signature", "handwriting"])
# ref = st.file_uploader("Reference Image", type=["png", "jpg", "jpeg", "tif"])
# test = st.file_uploader("Image to Verify", type=["png", "jpg", "jpeg", "tif"])
# threshold = st.slider("Decision threshold", 0.0, 2.0, 0.5, 0.01)

# if st.button("Verify") and ref and test:
#     with open("tmp_ref.png", "wb") as f:
#         f.write(ref.read())
#     with open("tmp_test.png", "wb") as f:
#         f.write(test.read())

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SiameseNetwork().to(device)

#     # pick model file
#     model_path = "models/siamese.pth" if task == "signature" else "models/siamese_handwriting.pth"
#     model.load_state_dict(torch.load(model_path, map_location=device))

#     dist, label = verify_pair(model, "tmp_ref.png", "tmp_test.png", device,
#                               threshold=threshold, task=task)

#     st.write(f"Distance: **{dist:.4f}**")
#     if ("Genuine" in label) or ("Same Writer" in label):
#         st.success(f"Result: **{label}**")
#     else:
#         st.error(f"Result: **{label}**")








from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
from train import SiameseNetwork
from verify import verify_pair
import shutil
import os

app = FastAPI()

# Allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all during testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models once at startup
models = {
    "signature": "models/siamese.pth",
    "handwriting": "models/siamese_handwriting.pth",
}
loaded_models = {}
for task, path in models.items():
    if os.path.exists(path):
        m = SiameseNetwork().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        loaded_models[task] = m


@app.post("/verify")
async def verify(
    ref: UploadFile = File(...),
    test: UploadFile = File(...),
    task: str = Form("signature"),
    threshold: float = Form(0.5)
):
    print("DEBUG =>", ref.filename, test.filename, task, threshold)
    ref_path, test_path = "tmp_ref.png", "tmp_test.png"

    # save uploaded images
    with open(ref_path, "wb") as f:
        shutil.copyfileobj(ref.file, f)
    with open(test_path, "wb") as f:
        shutil.copyfileobj(test.file, f)

    if task not in loaded_models:
        return {"error": f"Model for task '{task}' not loaded."}

    model = loaded_models[task]

    # get same output as streamlit
    dist, label = verify_pair(model, ref_path, test_path, device,
                              threshold=threshold, task=task)

    return {
        "distance": float(dist),
        "label": label   # <-- return the actual string (not boolean)
    }
