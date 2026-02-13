import base64
import json
import os
import shutil
import subprocess

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download


def render_clickable_image(image_path, link_url, width=100):
    """
    Generates a clickable image using HTML and Base64 encoding.
    """
    # 1. Read the image file and encode it to base64
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")

    # 2. Create the HTML string
    # target="_blank" opens the link in a new tab
    html_code = f"""
    <a href="{link_url}" target="_blank">
        <img src="data:image/png;base64,{data}" width="{width}" style="border-radius: 5px;">
    </a>
    """

    # 3. Render it
    st.markdown(html_code, unsafe_allow_html=True)


st.set_page_config(
    page_title="Prostate Scoring", page_icon="🩺", layout="wide", initial_sidebar_state="expanded"
)


@st.cache_data
def load_nrrd(file_path):
    """Load NRRD file and return data + header."""
    data, header = nrrd.read(file_path)
    return data, header


def display_slicer(scan_paths, mask_path=None, bboxes=None, title="Scan Viewer", key_suffix=""):
    """
    Displays slicer with Multi-Background Support, Mask Overlay, and Bounding Box Multiselect.

    Args:
        scan_paths: Dict of {Label: FilePath}. Example: {"T2W": "path/to/t2.nrrd", "ADC": "..."}
    """
    # 1. Layout: Image/Slider (Left) | Controls (Right)
    c_viewer, c_controls = st.columns([3, 1.5])

    # --- CONTROLS SECTION (Right Column) ---
    with c_controls:
        st.write(f"**{title} Controls**")

        # A. Background Selection
        # We assume the first key in the dict is the default
        available_scans = list(scan_paths.keys())
        selected_scan_name = st.radio(
            "Background Image", available_scans, index=0, key=f"bg_{key_suffix}"
        )
        current_file_path = scan_paths[selected_scan_name]

        # B. Lesion Selection (Multiselect)
        box_labels = []
        selected_labels = []
        if bboxes:
            box_labels = [f"Lesion {i + 1}" for i in range(len(bboxes))]
            st.write("---")  # Divider
            selected_labels = st.multiselect(
                "Select Lesions", options=box_labels, default=box_labels, key=f"multi_{key_suffix}"
            )

        # C. Toggles
        st.write("---")
        show_mask = False
        if mask_path and os.path.exists(mask_path):
            show_mask = st.checkbox("Show Mask Overlay", value=False, key=f"mk_{key_suffix}")

    # --- VIEWER SECTION (Left Column) ---
    with c_viewer:
        if not os.path.exists(current_file_path):
            st.error(f"File not found: {current_file_path}")
            return

        # Load the selected background image
        data, _ = load_nrrd(current_file_path)

        if len(data.shape) != 3:
            st.warning("Data is not 3D.")
            return

        total_slices = data.shape[2]

        # D. Slider Logic
        start_slice = total_slices // 2
        # Auto-jump logic: If exactly one lesion is selected, jump to it
        if len(selected_labels) == 1 and bboxes:
            idx = int(selected_labels[0].split(" ")[1]) - 1
            if 0 <= idx < len(bboxes):
                b = bboxes[idx]
                start_slice = int(b[2] + (b[5] // 2))
                start_slice = max(0, min(start_slice, total_slices - 1))

        slice_idx = st.slider(
            "Select Slice (Z-Axis)", 0, total_slices - 1, start_slice, key=f"sl_{key_suffix}"
        )

        # E. Plotting
        img_slice = data[:, :, slice_idx]

        # Normalize Image (0-1)
        img_slice = img_slice.astype(float)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img_slice, cmap="gray", origin="upper")

        # 1. Overlay Mask
        if show_mask:
            # Load mask on the fly (or cache it if slow)

            m_data, _ = load_nrrd(mask_path)
            # Check shape compatibility
            if m_data.shape == data.shape:
                mslice = m_data[:, :, slice_idx]
                overlay = np.ma.masked_where(mslice == 0, mslice)
                ax.imshow(overlay, cmap="Reds", alpha=0.5, origin="upper")
            else:
                # Fallback warning if mask dims don't match selected background
                # (Common if ADC resolution != T2 resolution)
                ax.text(5, 5, "Mask shape mismatch", color="red", fontsize=8)

        # 2. Overlay Bounding Boxes
        if bboxes:
            for i, box in enumerate(bboxes):
                label = f"Lesion {i + 1}"
                if label not in selected_labels:
                    continue

                bx, by, bz, bw, bh, bd = box

                # Visibility check
                if bz <= slice_idx < (bz + bd):
                    rect = patches.Rectangle(
                        (bx, by), bw, bh, linewidth=2, edgecolor="yellow", facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(bx, by - 5, f"L{i + 1}", color="yellow", fontsize=9, fontweight="bold")

        ax.axis("off")
        st.pyplot(fig, use_container_width=False)


@st.cache_resource
def download_all_models():
    # 1. Ensure the 'models' directory exists
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    for filename in FILENAMES:
        try:
            # 2. Download from Hugging Face (to cache)
            cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)

            # 3. Define where we want it to live locally
            destination_path = os.path.join(models_dir, filename)

            # 4. Copy only if it's not already there
            if not os.path.exists(destination_path):
                shutil.copy(cached_path, destination_path)

        except Exception as e:
            st.error(f"Failed to download {filename}: {e}")
            st.stop()


with st.container():
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_clickable_image(
            "deployment_images/logo1.png", "https://www.comfort-ai.eu/", width=220
        )
    with col2:
        render_clickable_image("deployment_images/logo2.png", "https://www.charite.de/", width=220)
    with col3:
        render_clickable_image("deployment_images/logo3.png", "https://mri.tum.de/de", width=220)
    with col4:
        render_clickable_image(
            "deployment_images/logo4.png", "https://ai-assisted-healthcare.com/", width=220
        )

st.write("")
st.write("")
st.title("PI-RADS and csPCa Risk Prediction from bpMRI")
# --- TRIGGER THE DOWNLOAD STARTUP ---
st.markdown(
    "💡 This application utilizes a weakly supervised, attention-based multiple-instance learning (MIL) model to predict scan-level PI-RADS scores and clinically significant prostate cancer (csPCa) risk from axial biparametric MRI (bpMRI) sequences (T2W, ADC, and DWI). Users may upload their own bpMRI scans as NRRD or select a provided sample case to evaluate the tool. Following inference, outcomes are detailed in the Results & Downloads section. The Visualization module allows users to inspect the prostate mask and the top five salient patches overlaid on the bpMRI sequences. The salient patches are displayed only when the predicted PI-RADS score is 3 or more. For execution details, refer to the log file; for methodology, please visit our [Project Page](https://anirudhbalaraman.github.io/WSAttention-Prostate/) or read the [Paper]. For more implementation details, check our [Github](https://github.com/anirudhbalaraman/WSAttention-Prostate/tree/main)"
)
st.markdown("***NOTE*** Required NRRD dimension format: Height x Width x Depth. ")

# --- CONSTANTS ---
REPO_ID = "anirudh0410/WSAttention-Prostate"
FILENAMES = ["pirads.pt", "prostate_segmentation_model.pt", "cspca_model.pth"]
with st.spinner("Initializing..."):
    download_all_models()
    st.success("Models ready!")

# --- CONFIGURATION ---
# Base paths
BASE_DIR = os.getcwd()
INPUT_BASE = os.path.join(BASE_DIR, "temp_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "temp_data", "processed")
SAMPLES_BASE_DIR = os.path.join(BASE_DIR, "dataset", "samples")
SAMPLE_CASES = {
    "Sample 1": {
        "path": os.path.join(SAMPLES_BASE_DIR, "sample1"),
        "files": {"t2": "t2.nrrd", "adc": "adc.nrrd", "dwi": "dwi.nrrd"},
    },
    "Sample 2": {
        "path": os.path.join(SAMPLES_BASE_DIR, "sample2"),
        "files": {"t2": "t2.nrrd", "adc": "adc.nrrd", "dwi": "dwi.nrrd"},
    },
    "Sample 3": {
        "path": os.path.join(SAMPLES_BASE_DIR, "sample3"),
        "files": {"t2": "t2.nrrd", "adc": "adc.nrrd", "dwi": "dwi.nrrd"},
    },
}

# Create specific sub-directories for each input type
# This ensures we pass a clean directory path to your script
T2_DIR = os.path.join(INPUT_BASE, "t2")
ADC_DIR = os.path.join(INPUT_BASE, "adc")
DWI_DIR = os.path.join(INPUT_BASE, "dwi")

# Ensure all folders exist
for path in [T2_DIR, ADC_DIR, DWI_DIR, OUTPUT_DIR]:
    os.makedirs(path, exist_ok=True)


# --- 1. DATA SOURCE SELECTION ---
with st.sidebar:
    st.header("Data Selection")
    # Dropdown to choose mode
    data_source = st.radio(
        "Choose Data Source:", ["Upload My Own Files", "Sample 1", "Sample 2", "Sample 3"], index=0
    )

# --- 2. INPUT HANDLING ---
t2_file = None
adc_file = None
dwi_file = None
is_demo_mode = data_source != "Upload My Own Files"
if is_demo_mode:
    # --- DEMO MODE LOGIC ---
    selected_sample = SAMPLE_CASES[data_source]
    st.info(f"👉 **Demo Mode Active:** Using {data_source}")

    # Verify files exist
    base_path = selected_sample["path"]
    f_names = selected_sample["files"]

    missing = []
    for _, fname in f_names.items():
        if not os.path.exists(os.path.join(base_path, fname)):
            missing.append(os.path.join(base_path, fname))

    if missing:
        st.error(f"Error: The following sample files are missing in the repo:\n{missing}")

    else:
        # Visual feedback
        c1, c2, c3 = st.columns(3)
        c1.success(f"T2: {f_names['t2']}")
        c2.success(f"ADC: {f_names['adc']}")
        c3.success(f"DWI: {f_names['dwi']}")

else:
    # --- UPLOAD MODE LOGIC ---

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
if "inference_done" not in st.session_state:
    st.session_state.inference_done = False
if "logs" not in st.session_state:
    st.session_state.logs = ""
ready_to_run = (not is_demo_mode and t2_file and adc_file and dwi_file) or is_demo_mode
if ready_to_run:
    if st.button("Run Inference", type="primary"):
        st.session_state.inference_done = False
        st.session_state.logs = ""
        # --- A. CLEANUP & SAVE ---
        # Clear old files to prevent mixing previous runs
        # (Optional but recommended for a clean state)
        for folder in [T2_DIR, ADC_DIR, DWI_DIR, OUTPUT_DIR]:
            for f in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, f)):
                    os.remove(os.path.join(folder, f))
                elif os.path.isdir(os.path.join(folder, f)):
                    shutil.rmtree(os.path.join(folder, f))

        if is_demo_mode:
            # Copy from the specific sample folder
            src = SAMPLE_CASES[data_source]
            shutil.copy(
                os.path.join(src["path"], src["files"]["t2"]), os.path.join(T2_DIR, "sample.nrrd")
            )
            shutil.copy(
                os.path.join(src["path"], src["files"]["adc"]), os.path.join(ADC_DIR, "sample.nrrd")
            )
            shutil.copy(
                os.path.join(src["path"], src["files"]["dwi"]), os.path.join(DWI_DIR, "sample.nrrd")
            )
            st.write(f"Loaded data from {data_source}...")

        else:
            # Save T2
            # We save it inside the T2_DIR folder
            with open(os.path.join(T2_DIR, t2_file.name), "wb") as f:
                shutil.copyfileobj(t2_file, f)

            # Save ADC
            with open(os.path.join(ADC_DIR, t2_file.name), "wb") as f:
                shutil.copyfileobj(adc_file, f)

            # Save DWI
            with open(os.path.join(DWI_DIR, t2_file.name), "wb") as f:
                shutil.copyfileobj(dwi_file, f)
            st.write("Uploaded files saved...")

        st.write("Starting Inference Pipeline...")

        # --- B. CONSTRUCT COMMAND ---
        # We pass the FOLDER paths, not file paths, matching your argument names
        command = [
            "python",
            "run_inference.py",
            "--t2_dir",
            T2_DIR,
            "--dwi_dir",
            DWI_DIR,
            "--adc_dir",
            ADC_DIR,
            "--output_dir",
            OUTPUT_DIR,
            "--project_dir",
            BASE_DIR,
        ]

        # DEBUG: Show the exact command being run (helpful for troubleshooting)
        st.code(" ".join(command), language="bash")

        # --- C. RUN SCRIPT ---
        with st.spinner("Running Inference... (This may take a moment)"):
            try:
                # Run the script and capture output
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                st.session_state.inference_done = True
                st.session_state.logs = result.stdout

            except subprocess.CalledProcessError as e:
                st.error("Script Execution Failed.")
                st.error("Error Output:")
                st.code(e.stderr)

                # --- D. SHOW OUTPUT FILES ---
if st.session_state.inference_done:
    st.success("Pipeline Execution Successful!")

    st.divider()
    with st.expander("📊 Results & Downloads", expanded=True):
        if st.session_state.get("logs"):  # Show Logs
            with st.expander("View Execution Logs"):
                st.code(st.session_state.logs)
        # List everything in the output directory

        output_files = os.listdir(OUTPUT_DIR)
        if output_files:
            for file_name in output_files:
                file_path = os.path.join(OUTPUT_DIR, file_name)
                if not os.path.isfile(file_path):
                    continue

                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"⬇️ Download {file_name}", data=f.read(), file_name=file_name
                    )
                    if file_name == "results.json":
                        with open(file_path) as f:
                            temp_data = json.load(f)
                        first_case = next(iter(temp_data.values()))
                        st.session_state.pirads = first_case.get("Predicted PIRAD Score")
                        st.session_state.risk = first_case.get("csPCa risk")
                        st.session_state.coords = first_case.get(
                            "Top left coordinate of top 5 patches(x,y,z)"
                        )

        else:
            st.warning("Script finished but no files were found in output_dir.")

        with st.expander("🩺 Results", expanded=True):
            if "risk" in st.session_state and "pirads" in st.session_state:
                # st.metric("csPCa Risk Score", f"{st.session_state.risk:.2f}")
                risk = st.session_state.get("risk")
                z = np.linspace(0, 1, 100).reshape(1, -1)  # 1 row, 100 columns
                col_chart, col_spacer = st.columns([1, 1])
                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Heatmap(
                            z=z,  # one row, two columns
                            x=np.linspace(0, 1, 100),  # 0 to 1 scale
                            y=[0, 1],
                            showscale=False,
                            colorscale="RdYlGn_r",
                            hoverinfo="none",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[risk],
                            y=[0.1],
                            mode="markers+text",
                            marker=dict(symbol="triangle-down", size=16, color="black"),
                            text=[f"csPCa Risk: {risk:.2f}"],
                            textposition="top center",
                            textfont=dict(color="black", size=16),
                            showlegend=False,
                            cliponaxis=False,
                        )
                    )

                    # Layout adjustments
                    fig.update_layout(
                        height=80,
                        margin=dict(l=20, r=20, t=20, b=25),
                        xaxis=dict(
                            range=[0, 1],
                            tickmode="array",
                            tickvals=[0, 1],
                            ticktext=["0", "1"],
                            showgrid=False,
                            ticks="outside",
                            ticklen=4,
                            tickfont=dict(size=16, color="black"),
                            ticklabelposition="inside bottom",
                            showline=False,
                            zeroline=False,
                            mirror=False,
                            side="bottom",
                        ),
                        yaxis=dict(
                            range=[0, 1], showticklabels=False, showgrid=False, showline=False
                        ),
                        plot_bgcolor="white",
                    )

                    st.plotly_chart(fig, use_container_width=False)

                pirads = st.session_state.get("pirads")
                score_config = {
                    2: {"bg": "#28a745", "text": "white"},  # Green
                    3: {"bg": "#ffc107", "text": "black"},  # Yellow
                    4: {"bg": "#fd7e14", "text": "white"},  # Orange
                    5: {"bg": "#dc3545", "text": "white"},  # Red
                }

                html_circles = ""

                for s in range(2, 6):
                    config = score_config[s]

                    # Define styles cleanly without newlines/indentation to prevent HTML errors
                    if s == int(pirads):
                        # Selected: Thick border, full opacity
                        border = "4px solid black"
                        opacity = "1.0"
                        transform = "scale(1.1)"
                        box_shadow = "0 4px 6px rgba(0,0,0,0.3)"
                    else:
                        # Unselected: Transparent border, low opacity
                        border = "4px solid transparent"
                        opacity = "0.3"
                        transform = "scale(1.0)"
                        box_shadow = "none"

                    # Build the div string
                    # distinct styling properties are joined by semicolons
                    html_circles += f"""
                    <div style="
                        width: 60px;
                        height: 60px;
                        background-color: {config["bg"]};
                        color: {config["text"]};
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-size: 24px;
                        font-weight: bold;
                        font-family: Arial, sans-serif;
                        margin-right: 15px;
                        border: {border};
                        opacity: {opacity};
                        transform: {transform};
                        box-shadow: {box_shadow};">
                        {s}
                    </div>
                    """

                # Display Container
                st.markdown(f"### PI-RADS Score: {pirads}")
                st.markdown(
                    f"""
                    <div style="display: flex; flex-direction: row; align-items: center; padding: 10px 0;">
                        {html_circles}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Results not available.")

    with st.expander("Visualisation", expanded=True):
        t2_vis_path = None
        dwi_vis_path = None
        adc_vis_path = None
        mask_vis_path = None

        t2_vis_dir = os.path.join(OUTPUT_DIR, "t2_registered")
        if os.path.exists(t2_vis_dir) and len(os.listdir(t2_vis_dir)) > 0:
            files_in_dir = os.listdir(t2_vis_dir)[0]
            t2_vis_path = os.path.join(t2_vis_dir, files_in_dir)

        adc_vis_dir = os.path.join(OUTPUT_DIR, "ADC_registered")
        if os.path.exists(adc_vis_dir) and len(os.listdir(adc_vis_dir)) > 0:
            files_in_dir = os.listdir(adc_vis_dir)[0]
            adc_vis_path = os.path.join(adc_vis_dir, files_in_dir)

        dwi_vis_dir = os.path.join(OUTPUT_DIR, "DWI_registered")
        if os.path.exists(dwi_vis_dir) and len(os.listdir(dwi_vis_dir)) > 0:
            files_in_dir = os.listdir(dwi_vis_dir)[0]
            dwi_vis_path = os.path.join(dwi_vis_dir, files_in_dir)

        mask_vis_dir = os.path.join(OUTPUT_DIR, "prostate_mask")
        if os.path.exists(mask_vis_dir) and len(os.listdir(mask_vis_dir)) > 0:
            files_in_maskdir = os.listdir(mask_vis_dir)[0]
            mask_vis_path = os.path.join(mask_vis_dir, files_in_maskdir)
            print("mask_vis_path")
        else:
            print("No mask dir")

        roi_bbox = None
        if "coords" in st.session_state:
            detected_boxes = []
            for i in st.session_state.coords:
                indi_box = [i[1], i[0], i[2], 64, 64, 3]
                detected_boxes.append(indi_box)

        scan_dict = {}
        if t2_vis_path and os.path.exists(t2_vis_path):
            scan_dict["T2W"] = t2_vis_path
        if adc_vis_path and os.path.exists(adc_vis_path):
            scan_dict["ADC"] = adc_vis_path
        if dwi_vis_path and os.path.exists(dwi_vis_path):
            scan_dict["DWI"] = dwi_vis_path

        if scan_dict and st.session_state.pirads > 2:
            display_slicer(
                scan_paths=scan_dict,  # <--- Pass the Dict here
                mask_path=mask_vis_path,
                bboxes=detected_boxes,
                title="Salient Patch Viewer",
                key_suffix="main_viz",
            )
        elif scan_dict:
            display_slicer(
                scan_paths=scan_dict,  # <--- Pass the Dict here
                mask_path=mask_vis_path,
                bboxes=None,
                title="Salient Patch Viewer",
                key_suffix="main_viz",
            )
