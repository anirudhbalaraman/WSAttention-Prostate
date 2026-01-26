import streamlit as st
import subprocess
import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "anirudh0410/WSAttention-Prostate" # <--- UPDATE THIS
FILENAME = ["pirads.pt", "prostate_segmentation_model.pt", "cspca_model.pth"] # The name of the file inside Hugging Face

@st.cache_resource
def get_model_path(filename):
    # 1. Download the file from Hugging Face
    # This downloads it to a cache folder on the machine
    cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    
    # 2. Copy it to the current directory
    # Your run_inference.py likely expects 'saved_model.pth' to be right here
    if not os.path.exists(filename):
        shutil.copy(cached_path, os.path.join(os.getcwd(), 'models',filename))

    return filename

# --- TRIGGER DOWNLOAD ---
# Run this immediately when the app starts
try:
    with st.spinner("Fetching model from Hugging Face..."):
        for i in FILENAME:
            local_model_path = get_model_path(i)
            st.success("Model ready!")
except Exception as e:
    st.error(f"Error downloading model: {e}")
    st.stop()

# --- CONFIGURATION ---
# Base paths
BASE_DIR = os.getcwd()
INPUT_BASE = os.path.join(BASE_DIR, "temp_data" )
OUTPUT_DIR = os.path.join(BASE_DIR, "temp_data", "processed")

# Create specific sub-directories for each input type
# This ensures we pass a clean directory path to your script
T2_DIR = os.path.join(INPUT_BASE, "t2")
ADC_DIR = os.path.join(INPUT_BASE, "adc")
DWI_DIR = os.path.join(INPUT_BASE, "dwi")

# Ensure all folders exist
for path in [T2_DIR, ADC_DIR, DWI_DIR, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)

st.title("Model Inference")
st.markdown("### Upload your T2W, ADC, and DWI scans")

# --- 1. UI: THREE UPLOADERS ---
col1, col2, col3 = st.columns(3)

with col1:
    t2_file = st.file_uploader("Upload T2W (NRRD)", type=["nrrd"])
with col2:
    adc_file = st.file_uploader("Upload ADC (NRRD)", type=["nrrd"])
with col3:
    dwi_file = st.file_uploader("Upload DWI (NRRD)", type=["nrrd"])

# --- 2. EXECUTION LOGIC ---
if t2_file and adc_file and dwi_file:
    st.success("Files ready.")
    
    if st.button("Run Inference"):
        # --- A. CLEANUP & SAVE ---
        # Clear old files to prevent mixing previous runs
        # (Optional but recommended for a clean state)
        for folder in [T2_DIR, ADC_DIR, DWI_DIR, OUTPUT_DIR]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

        # Save T2
        # We save it inside the T2_DIR folder
        with open(os.path.join(T2_DIR, t2_file.name), "wb") as f:
            shutil.copyfileobj(t2_file, f)
            
        # Save ADC
        with open(os.path.join(ADC_DIR, adc_file.name), "wb") as f:
            shutil.copyfileobj(adc_file, f)
            
        # Save DWI
        with open(os.path.join(DWI_DIR, dwi_file.name), "wb") as f:
            shutil.copyfileobj(dwi_file, f)

        st.write("Files saved. Starting pipeline...")

        # --- B. CONSTRUCT COMMAND ---
        # We pass the FOLDER paths, not file paths, matching your argument names
        command = [
            "python", "run_inference.py",
            "--t2_dir", T2_DIR,
            "--dwi_dir", DWI_DIR,
            "--adc_dir", ADC_DIR,
            "--output_dir", OUTPUT_DIR,
            "--project_dir", BASE_DIR
        ]
        
        # DEBUG: Show the exact command being run (helpful for troubleshooting)
        st.code(" ".join(command), language="bash")

        # --- C. RUN SCRIPT ---
        with st.spinner("Running Inference... (This may take a moment)"):
            try:
                # Run the script and capture output
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                st.success("Pipeline Execution Successful!")
                
                # Show Logs
                with st.expander("View Execution Logs"):
                    st.code(result.stdout)
                
                # --- D. SHOW OUTPUT FILES ---
                st.subheader("Results & Downloads")
                
                # List everything in the output directory
                if os.path.exists(OUTPUT_DIR):
                    output_files = os.listdir(OUTPUT_DIR)
                    
                    if output_files:
                        for file_name in output_files:
                            file_path = os.path.join(OUTPUT_DIR, file_name)
                            
                            # Skip directories, show download buttons for files
                            if os.path.isfile(file_path):
                                with open(file_path, "rb") as f:
                                    st.download_button(
                                        label=f"Download {file_name}",
                                        data=f,
                                        file_name=file_name
                                    )
                    else:
                        st.warning("Script finished but no files were found in output_dir.")
                        
            except subprocess.CalledProcessError as e:
                st.error("Script Execution Failed.")
                st.error("Error Output:")
                st.code(e.stderr)