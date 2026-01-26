import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import nrrd
import matplotlib.pyplot as plt
import tempfile
import os

# --- 1. IMPORT CUSTOM SCRIPTS ---
try:
    from model_definition import MyModelClass
    # Your preprocess function should now likely accept a LIST of arrays or a stacked array
    from inference_utils import preprocess_multimodal 
except ImportError:
    st.warning("Could not import custom modules.")

# --- 2. CONFIGURATION ---
MODEL_PATH = 'saved_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    try:
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

# --- 4. APP INTERFACE ---
st.title("Multi-Modal Medical Inference")
st.write("Upload exactly 3 NRRD files (e.g., T1, T2, FLAIR) to generate a prediction.")

# Update: accept_multiple_files=True
uploaded_files = st.file_uploader("Choose 3 NRRD files...", type=["nrrd"], accept_multiple_files=True)

# LOGIC: Only proceed if exactly 3 files are present
if uploaded_files:
    if len(uploaded_files) != 3:
        st.warning(f"Please upload exactly 3 files. You currently have {len(uploaded_files)}.")
    else:
        st.success("3 Files Uploaded. Processing...")
        
        # Sort files by name to ensure consistent order (e.g., file_01, file_02, file_03)
        # This is CRITICAL if your model expects channels in a specific order.
        uploaded_files.sort(key=lambda x: x.name)
        
        scan_data_list = []
        temp_paths = []

        try:
            # --- A. Read all 3 files ---
            # We create columns to show previews side-by-side
            cols = st.columns(3)
            
            for idx, file in enumerate(uploaded_files):
                # Save to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nrrd") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                    temp_paths.append(tmp_path)
                
                # Read NRRD
                data, header = nrrd.read(tmp_path)
                scan_data_list.append(data)
                
                # Visualize Middle Slice in the respective column
                with cols[idx]:
                    st.caption(file.name)
                    mid_slice = data.shape[2] // 2 if data.ndim == 3 else 0
                    
                    fig, ax = plt.subplots()
                    # Show slice (assuming 3D data: H, W, D)
                    ax.imshow(data[:, :, mid_slice], cmap="gray")
                    ax.axis("off")
                    st.pyplot(fig)

            # --- B. Combine/Stack Data ---
            if st.button("Run Prediction"):
                st.write("Merging channels and analyzing...")
                
                # STACKING LOGIC:
                # We assume the 3 files represent 3 channels.
                # If each data is (H, W, D), result is (3, H, W, D)
                # We stack along a new dimension (axis 0)
                stacked_volume = np.stack(scan_data_list, axis=0) 
                
                # --- C. Preprocessing ---
                # Pass this (3, ...) array to your pipeline
                input_tensor = preprocess_multimodal(stacked_volume)

                # Ensure Batch Dimension (1, 3, D, H, W)
                if isinstance(input_tensor, torch.Tensor):
                    if input_tensor.ndim == 4: # (3, D, H, W) -> (1, 3, D, H, W)
                        input_tensor = input_tensor.unsqueeze(0)
                    input_tensor = input_tensor.to(DEVICE)

                # --- D. Inference ---
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = F.softmax(output, dim=1)
                    confidence, predicted_class_idx = torch.max(probabilities, 1)

                st.success("Done!")
                st.metric("Prediction Class", predicted_class_idx.item())
                st.metric("Confidence", f"{confidence.item()*100:.2f}%")

        except Exception as e:
            st.error(f"Error during processing: {e}")
            
        finally:
            # Cleanup temp files
            for p in temp_paths:
                if os.path.exists(p):
                    os.remove(p)