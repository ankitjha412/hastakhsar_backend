# import streamlit as st
# import torch
# from verify import SiameseNetwork, verify_signature

# st.title("✍️ Handwriting Forgery Detector")

# ref = st.file_uploader("Reference Signature (genuine)", type=["png", "jpg", "jpeg"])
# test = st.file_uploader("Signature to verify", type=["png", "jpg", "jpeg"])
# threshold = st.slider("Decision threshold (lower→more strict)", 0.0, 2.0, 0.5, 0.01)

# if st.button("Verify") and ref and test:
#     with open("tmp_ref.png", "wb") as f:
#         f.write(ref.read())
#     with open("tmp_test.png", "wb") as f:
#         f.write(test.read())

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = SiameseNetwork().to(device)
#     model.load_state_dict(torch.load("models/siamese.pth", map_location=device))
    
#     dist, label = verify_signature(model, "tmp_ref.png", "tmp_test.png", device, threshold=threshold)

#     st.write(f"Distance: **{dist:.4f}**")
#     if label == "Genuine":
#         st.success(f"Result: **{label}**")
#     else:
#         st.error(f"Result: **{label}**")





import streamlit as st
import torch
from train import SiameseNetwork
from verify import verify_pair

st.title("✍️ Handwriting & Signature Forgery Detector")

task = st.radio("Select Task", ["signature", "handwriting"])
ref = st.file_uploader("Reference Image", type=["png", "jpg", "jpeg", "tif"])
test = st.file_uploader("Image to Verify", type=["png", "jpg", "jpeg", "tif"])
threshold = st.slider("Decision threshold", 0.0, 2.0, 0.5, 0.01)

if st.button("Verify") and ref and test:
    with open("tmp_ref.png", "wb") as f:
        f.write(ref.read())
    with open("tmp_test.png", "wb") as f:
        f.write(test.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)

    # pick model file
    model_path = "models/siamese.pth" if task == "signature" else "models/siamese_handwriting.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    dist, label = verify_pair(model, "tmp_ref.png", "tmp_test.png", device,
                              threshold=threshold, task=task)

    st.write(f"Distance: **{dist:.4f}**")
    if ("Genuine" in label) or ("Same Writer" in label):
        st.success(f"Result: **{label}**")
    else:
        st.error(f"Result: **{label}**")

