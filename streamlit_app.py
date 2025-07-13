"""
Streamlit App for Neural Data Analysis Pipeline
"""

import streamlit as st
st.markdown("""
    <style>
    /* Make all text, labels, and help text white for dark theme */
    label, .stSelectbox label, .stRadio label, .st-emotion-cache-1kyxreq, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1r6slb0 {
        color: #fff !important;
        font-weight: 500 !important;
    }
    </style>
""", unsafe_allow_html=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import json
import tempfile
import os
from neural_backend import NeuralDataAnalyzer
from simple_ai_assistant import ai_assistant

# Import advanced modules
try:
    from realtime_eeg import create_realtime_dashboard
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False

try:
    from advanced_visualization import create_advanced_visualization_interface
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

try:
    from advanced_analysis import create_advanced_analysis_interface
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# Import data pipeline modules
try:
    from data_pipeline import DataPipeline, PipelineConfig, BatchProcessor, DataValidator
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from workflow_automation import WorkflowManager, WorkflowExecutor, ReproducibilityManager
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False

try:
    from advanced_data_management import BIDSManager, DatabaseManager, DataOrganizer, BIDSParticipant
    DATA_MANAGEMENT_AVAILABLE = True
except ImportError:
    DATA_MANAGEMENT_AVAILABLE = False

try:
    from enhanced_ux import (
        KeyboardShortcutManager, DashboardManager, SessionManager, 
        PerformanceOptimizer, AccessibilityManager, render_enhanced_sidebar,
        render_keyboard_shortcuts_help, render_dashboard_customization,
        initialize_enhanced_ux
    )
    UX_AVAILABLE = True
except ImportError:
    UX_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Neural Data Analysis Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme configuration
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # Default to dark theme

# Dynamic CSS: Force dark theme with white text everywhere
def get_theme_css(theme=None):
    return """
<style>
    body, .stApp, .main, .block-container, .css-1d391kg, .css-fg4lbf, .sidebar-content, .stSidebar, .stSidebarContent, .css-1lcbmhc {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    .stMarkdown, .stText, .stHeader, .stSubheader, .stCaption, .stDataFrame, .stTable, .stMetric, .stInfo, .stSuccess, .stWarning, .stError, .stException, .stAlert, .stExpander, .stRadio, .stSelectbox, .stNumberInput, .stSlider, .stCheckbox, .stButton, .stDownloadButton, .stFileUploader, .stTextInput, .stTextArea, .stCode, .stJson, .stTabs, .stTab, .stForm, .stFormSubmitButton, .stColorPicker, .stDateInput, .stTimeInput, .stMultiSelect, .stProgress, .stSpinner, .stImage, .stAudio, .stVideo, .stPlotlyChart, .stPyplotChart, .stVegaLiteChart, .stAltairChart, .stBokehChart, .stDeckGlChart, .stGraphvizChart, .stMap, .stAgGrid, .stAgGridTable, .stAgGridCell, .stAgGridHeader, .stAgGridFooter, .stAgGridPagination, .stAgGridToolbar, .stAgGridFilter, .stAgGridSort, .stAgGridGroup, .stAgGridRow, .stAgGridColumn, .stAgGridColumnHeader, .stAgGridColumnFooter, .stAgGridColumnGroup, .stAgGridColumnGroupHeader, .stAgGridColumnGroupFooter, .stAgGridColumnGroupToolbar, .stAgGridColumnGroupFilter, .stAgGridColumnGroupSort, .stAgGridColumnGroupGroup, .stAgGridColumnGroupRow, .stAgGridColumnGroupColumn, .stAgGridColumnGroupColumnHeader, .stAgGridColumnGroupColumnFooter, .stAgGridColumnGroupColumnToolbar, .stAgGridColumnGroupColumnFilter, .stAgGridColumnGroupColumnSort, .stAgGridColumnGroupColumnGroup, .stAgGridColumnGroupColumnRow, .stAgGridColumnGroupColumnColumn, .stAgGridColumnGroupColumnColumnHeader, .stAgGridColumnGroupColumnColumnFooter, .stAgGridColumnGroupColumnColumnToolbar, .stAgGridColumnGroupColumnColumnFilter, .stAgGridColumnGroupColumnColumnSort, .stAgGridColumnGroupColumnColumnGroup, .stAgGridColumnGroupColumnColumnRow, .stAgGridColumnGroupColumnColumnColumn, .stAgGridColumnGroupColumnColumnColumnHeader, .stAgGridColumnGroupColumnColumnColumnFooter, .stAgGridColumnGroupColumnColumnColumnToolbar, .stAgGridColumnGroupColumnColumnColumnFilter, .stAgGridColumnGroupColumnColumnColumnSort, .stAgGridColumnGroupColumnColumnColumnGroup, .stAgGridColumnGroupColumnColumnColumnRow, .stAgGridColumnGroupColumnColumnColumnColumn {
        color: #fafafa !important;
    }
    .stButton>button, .stDownloadButton>button {
        background-color: #1f77b4;
        color: #fafafa;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: 1px solid #4a4a4a;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #165b87;
        color: #fafafa;
        border-color: #666666;
    }
    .info-box {
        background-color: #1e3a5f;
        color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #1e4d2e;
        color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }
    .js-plotly-plot {
        background-color: #0e1117 !important;
    }
    .end-watermark {
        color: rgba(250, 250, 250, 0.4);
        font-size: 14px;
        font-family: sans-serif;
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        padding-bottom: 20px;
        border-top: 1px solid rgba(250, 250, 250, 0.1);
    }
    .dataframe {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
    .css-1wivap2 {
        background-color: #262730 !important;
        color: #fafafa !important;
    }
</style>
"""

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# Set matplotlib theme
def set_matplotlib_theme(theme):
    """Set matplotlib theme based on app theme"""
    if theme == 'dark':
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#0e1117'
        plt.rcParams['axes.facecolor'] = '#0e1117'
        plt.rcParams['text.color'] = '#fafafa'
        plt.rcParams['axes.labelcolor'] = '#fafafa'
        plt.rcParams['xtick.color'] = '#fafafa'
        plt.rcParams['ytick.color'] = '#fafafa'
    else:
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = '#ffffff'
        plt.rcParams['axes.facecolor'] = '#ffffff'
        plt.rcParams['text.color'] = '#262730'
        plt.rcParams['axes.labelcolor'] = '#262730'
        plt.rcParams['xtick.color'] = '#262730'
        plt.rcParams['ytick.color'] = '#262730'

# Apply matplotlib theme
set_matplotlib_theme(st.session_state.theme)

# Initialize session state variables
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = NeuralDataAnalyzer()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_preprocessed' not in st.session_state:
    st.session_state.data_preprocessed = False
if 'epochs_created' not in st.session_state:
    st.session_state.epochs_created = False
if 'tfr_computed' not in st.session_state:
    st.session_state.tfr_computed = False
if 'features_extracted' not in st.session_state:
    st.session_state.features_extracted = False
if 'pac_computed' not in st.session_state:
    st.session_state.pac_computed = False
if 'ml_completed' not in st.session_state:
    st.session_state.ml_completed = False

# --- Sidebar ---
st.sidebar.title("🧠 Neural Analysis App")

# Theme Switcher
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Theme Settings")
theme_option = st.sidebar.selectbox(
    "Choose Theme:",
    ["Dark", "Light"],
    index=0 if st.session_state.theme == 'dark' else 1,
    key="theme_selector"
)

# Update theme if changed
new_theme = 'dark' if theme_option == "Dark" else 'light'
if new_theme != st.session_state.theme:
    st.session_state.theme = new_theme
    st.rerun()  # Rerun to apply new theme

st.sidebar.markdown("---")

# Advanced Features Section
st.sidebar.subheader("🚀 Advanced Features")
advanced_feature = st.sidebar.selectbox(
    "Select Advanced Feature:",
    ["Main Analysis", "Real-Time EEG", "Advanced Visualization", "Advanced Analysis"],
    key="advanced_feature_selector"
)

st.sidebar.markdown("---")

# Section 1: Data Loading
st.sidebar.header("1. Load Data")
data_source = st.sidebar.radio("Choose data source:", ("Sample Data", "Upload File"), key="data_source_radio")

if data_source == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Upload EEG/LFP file (.edf, .fif, .set)", type=["edf", "fif", "set"])
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_filepath = tmp_file.name
        
        file_extension = os.path.splitext(uploaded_file.name)[1].lower().lstrip('.')
        
        if st.sidebar.button("Load Uploaded Data", key="load_uploaded_btn"):
            with st.spinner(f"Loading {uploaded_file.name}..."):
                success = st.session_state.analyzer.load_data(temp_filepath, file_type=file_extension)
                if success:
                    st.session_state.data_loaded = True
                    st.sidebar.success("Data loaded successfully!")
                    # Clean up temp file immediately after loading
                    os.unlink(temp_filepath)
                else:
                    st.session_state.data_loaded = False
                    st.sidebar.error("Failed to load data.")
elif data_source == "Sample Data":
    if st.sidebar.button("Load MNE Sample Data", key="load_sample_btn"):
        with st.spinner("Loading sample data..."):
            success = st.session_state.analyzer.load_sample_data()
            if success:
                st.session_state.data_loaded = True
                st.sidebar.success("MNE Sample Data loaded!")
            else:
                st.session_state.data_loaded = False
                st.sidebar.error("Failed to load sample data.")

# Section 2: Preprocessing
st.sidebar.header("2. Preprocessing")
if st.session_state.data_loaded:
    l_freq = st.sidebar.number_input("Lower cut-off frequency (Hz)", value=1.0, min_value=0.0, max_value=20.0, key="l_freq")
    h_freq = st.sidebar.number_input("Upper cut-off frequency (Hz)", value=40.0, min_value=10.0, max_value=200.0, key="h_freq")
    resample_freq = st.sidebar.number_input("Resample frequency (Hz) (0 for no resample)", value=0, min_value=0, max_value=1000, key="resample_freq")
    if resample_freq == 0: resample_freq = None
    notch_freq = st.sidebar.number_input("Notch filter (e.g. 50 or 60 Hz) (0 for none)", value=0, min_value=0, max_value=200, key="notch_freq")
    if notch_freq == 0: notch_freq = None

    if st.sidebar.button("Preprocess Data", key="preprocess_btn"):
        with st.spinner("Preprocessing data..."):
            success = st.session_state.analyzer.preprocess_data(l_freq=l_freq, h_freq=h_freq, resample_freq=resample_freq, notch_freq=notch_freq)
            if success:
                st.session_state.data_preprocessed = True
                st.sidebar.success("Data preprocessed!")
            else:
                st.session_state.data_preprocessed = False
                st.sidebar.error("Preprocessing failed.")
else:
    st.sidebar.info("Load data first to enable preprocessing.")

# Section 3: Epoching
st.sidebar.header("3. Create Epochs")
if st.session_state.data_preprocessed:
    tmin = st.sidebar.number_input("Epoch start time (s)", value=-0.5, key="tmin")
    tmax = st.sidebar.number_input("Epoch end time (s)", value=1.0, key="tmax")
    baseline_min = st.sidebar.number_input("Baseline start time (s)", value=-0.2, key="baseline_min")
    baseline_max = st.sidebar.number_input("Baseline end time (s)", value=0.0, key="baseline_max")

    if st.sidebar.button("Create Epochs", key="create_epochs_btn"):
        with st.spinner("Creating epochs..."):
            success = st.session_state.analyzer.create_epochs(tmin=tmin, tmax=tmax, baseline=(baseline_min, baseline_max))
            if success:
                st.session_state.epochs_created = True
                st.sidebar.success("Epochs created!")
            else:
                st.session_state.epochs_created = False
                st.sidebar.error("Failed to create epochs.")
else:
    st.sidebar.info("Preprocess data first to enable epoching.")

# Section 4: Time-Frequency Analysis
st.sidebar.header("4. Time-Frequency Analysis")
if st.session_state.epochs_created:
    freq_min = st.sidebar.number_input("Min frequency (Hz)", value=4.0, min_value=1.0, max_value=10.0, key="freq_min")
    freq_max = st.sidebar.number_input("Max frequency (Hz)", value=40.0, min_value=10.0, max_value=100.0, key="freq_max")
    n_freqs = st.sidebar.number_input("Number of frequencies", value=20, min_value=5, max_value=100, key="n_freqs")
    
    if st.sidebar.button("Compute TFR", key="compute_tfr_btn"):
        with st.spinner("Computing Time-Frequency Representation..."):
            freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
            success = st.session_state.analyzer.time_frequency_analysis(freqs=freqs)
            if success:
                st.session_state.tfr_computed = True
                st.sidebar.success("TFR computed!")
            else:
                st.session_state.tfr_computed = False
                st.sidebar.error("Failed to compute TFR. Check TFR data dimensions.")
else:
    st.sidebar.info("Create epochs first to enable TFR analysis.")

# Section 5: Feature Extraction
st.sidebar.header("5. Feature Extraction (for ML)")
if st.session_state.tfr_computed:
    st.sidebar.subheader("Frequency Bands for Features")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        theta_min = st.number_input("Theta min (Hz)", value=4.0, key="theta_min")
        alpha_min = st.number_input("Alpha min (Hz)", value=8.0, key="alpha_min")
        beta_min = st.number_input("Beta min (Hz)", value=12.0, key="beta_min")
        gamma_min = st.number_input("Gamma min (Hz)", value=30.0, key="gamma_min")
    with col2:
        theta_max = st.number_input("Theta max (Hz)", value=8.0, key="theta_max")
        alpha_max = st.number_input("Alpha max (Hz)", value=12.0, key="alpha_max")
        beta_max = st.number_input("Beta max (Hz)", value=30.0, key="beta_max")
        gamma_max = st.number_input("Gamma max (Hz)", value=40.0, key="gamma_max")

    freq_bands = {
        'theta': (theta_min, theta_max),
        'alpha': (alpha_min, alpha_max),
        'beta': (beta_min, beta_max),
        'gamma': (gamma_min, gamma_max)
    }
    
    time_win_start = st.sidebar.number_input("Feature time window start (s)", value=0.0, key="time_win_start")
    time_win_end = st.sidebar.number_input("Feature time window end (s)", value=0.5, key="time_win_end")

    if st.sidebar.button("Extract Power Features", key="extract_features_btn"):
        with st.spinner("Extracting features..."):
            success = st.session_state.analyzer.extract_power_features(freq_bands=freq_bands, time_window=(time_win_start, time_win_end))
            if success:
                st.session_state.features_extracted = True
                st.sidebar.success("Features extracted!")
            else:
                st.session_state.features_extracted = False
                st.sidebar.error("Failed to extract features. Check TFR data dimensions (should be 4D).")
else:
    st.sidebar.info("Compute TFR first to enable feature extraction.")

# Section 6: PAC Analysis
st.sidebar.header("6. PAC Analysis")
if st.session_state.epochs_created: # PAC uses epochs data directly
    st.sidebar.subheader("Phase-Amplitude Coupling Frequencies")
    phase_min = st.sidebar.number_input("Phase freq min (Hz)", value=4.0, min_value=1.0, max_value=10.0, key="pac_phase_min")
    phase_max = st.sidebar.number_input("Phase freq max (Hz)", value=8.0, min_value=6.0, max_value=15.0, key="pac_phase_max")

    amp_min = st.sidebar.number_input("Amplitude freq min (Hz)", value=30.0, min_value=20.0, max_value=50.0, key="pac_amp_min")
    amp_max = st.sidebar.number_input("Amplitude freq max (Hz)", value=40.0, min_value=30.0, max_value=80.0, key="pac_amp_max")
    
    # Add PAC method selection
    pac_method = st.sidebar.selectbox(
        "PAC Method:",
        options=['tort', 'canolty', 'ozkurt', 'plv'],
        index=0,
        help="Tort: Modulation Index, Canolty: Mean Vector Length, Ozkurt: Phase-Locking Value, PLV: Phase-Locking Value approach"
    )

    if st.sidebar.button("Compute PAC", key="compute_pac_btn"):
        with st.spinner("Computing Phase-Amplitude Coupling..."):
            success = st.session_state.analyzer.compute_phase_amplitude_coupling(
                low_freq=(phase_min, phase_max), 
                high_freq=(amp_min, amp_max),
                method=pac_method
            )
            if success:
                st.session_state.pac_computed = True
                st.sidebar.success("PAC computed!")
            else:
                st.session_state.pac_computed = False
                st.sidebar.error("Failed to compute PAC.")
else:
    st.sidebar.info("Create epochs first to enable PAC analysis.")

# Section 7: Machine Learning Classification
st.sidebar.header("7. ML Classification")
if st.session_state.features_extracted and st.session_state.epochs_created:
    test_size = st.sidebar.slider("Test size for ML (0.1 - 0.5)", value=0.3, min_value=0.1, max_value=0.5, step=0.05, key="test_size")
    
    if st.sidebar.button("Run Classification", key="run_ml_btn"):
        with st.spinner("Running classification..."):
            try:
                success = st.session_state.analyzer.classify_conditions(test_size=test_size)
                if success:
                    st.session_state.ml_completed = True
                    st.sidebar.success("Classification completed!")
                else:
                    st.session_state.ml_completed = False
                    st.sidebar.error("Classification failed.")
            except ValueError as e:
                st.session_state.ml_completed = False # Ensure status is reset on error
                st.sidebar.error(f"Classification Error: {e}. Ensure features and labels match (likely TFR issue).")
            except Exception as e:
                st.session_state.ml_completed = False # Ensure status is reset on error
                st.sidebar.error(f"An unexpected error occurred during classification: {e}")
else:
    st.sidebar.info("Extract features and create epochs first to enable ML classification.")

# --- Artifact Detection & Removal ---
st.sidebar.header("🧹 Artifact Detection & Removal")

# ICA controls
if st.session_state.data_loaded:
    st.sidebar.subheader("ICA Artifact Removal")
    if st.sidebar.button("Run ICA", key="run_ica_btn"):
        with st.spinner("Running ICA..."):
            try:
                ica = st.session_state.analyzer.run_ica()
                st.session_state.ica_run = True
                st.sidebar.success("ICA fitting complete!")
            except Exception as e:
                st.sidebar.error(f"ICA error: {e}")
    if hasattr(st.session_state.analyzer, 'ica'):
        st.sidebar.info("ICA has been run. You can plot and exclude components.")
        # Plot ICA components (topomaps)
        if st.button("Show ICA Components (Topomap)", key="show_ica_topo"):
            try:
                fig = st.session_state.analyzer.ica.plot_components(show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting ICA components: {e}")
        # Select components to exclude
        n_comp = st.session_state.analyzer.ica.n_components_
        exclude = st.sidebar.multiselect(
            "Select ICA components to exclude:",
            options=list(range(n_comp)),
            key="ica_exclude_select"
        )
        if st.sidebar.button("Apply ICA", key="apply_ica_btn"):
            with st.spinner("Applying ICA..."):
                try:
                    st.session_state.analyzer.apply_ica(exclude=exclude)
                    st.sidebar.success(f"ICA applied. Excluded: {exclude}")
                except Exception as e:
                    st.sidebar.error(f"ICA apply error: {e}")

# Amplitude thresholding
if st.session_state.epochs_created:
    st.sidebar.subheader("Amplitude Thresholding")
    threshold = st.sidebar.number_input(
        "Artifact threshold (µV, peak-to-peak)",
        value=150.0, min_value=10.0, max_value=1000.0, step=10.0, key="artifact_thresh_input"
    )
    if st.sidebar.button("Detect Bad Epochs", key="detect_bad_epochs_btn"):
        with st.spinner("Detecting artifacts..."):
            try:
                bad_epochs = st.session_state.analyzer.detect_artifacts(threshold=threshold*1e-6)
                st.session_state.bad_epochs = bad_epochs
                st.sidebar.success(f"Detected {len(bad_epochs)} bad epochs.")
            except Exception as e:
                st.sidebar.error(f"Artifact detection error: {e}")
    # Manual marking
    st.sidebar.subheader("Manual Marking")
    # Bad channels
    all_channels = st.session_state.analyzer.raw.ch_names if st.session_state.analyzer.raw else []
    bad_channels = st.sidebar.multiselect(
        "Mark bad channels:",
        options=all_channels,
        key="bad_channels_select"
    )
    if st.sidebar.button("Apply Bad Channels", key="apply_bad_channels_btn"):
        try:
            st.session_state.analyzer.mark_bad_channels(bad_channels)
            st.sidebar.success(f"Marked bad channels: {bad_channels}")
        except Exception as e:
            st.sidebar.error(f"Bad channel marking error: {e}")
    # Bad epochs
    n_epochs = len(st.session_state.analyzer.epochs) if st.session_state.analyzer.epochs else 0
    bad_epochs_manual = st.sidebar.text_input(
        "Mark bad epochs (comma-separated indices):",
        value="",
        key="bad_epochs_input"
    )
    if st.sidebar.button("Apply Bad Epochs", key="apply_bad_epochs_btn"):
        try:
            if bad_epochs_manual.strip():
                bad_idx = [int(idx.strip()) for idx in bad_epochs_manual.split(",") if idx.strip().isdigit()]
                st.session_state.analyzer.mark_bad_epochs(bad_idx)
                st.sidebar.success(f"Dropped bad epochs: {bad_idx}")
            else:
                st.sidebar.info("No epochs specified.")
        except Exception as e:
            st.sidebar.error(f"Bad epoch marking error: {e}")




# --- Main Panel ---
st.markdown("<h1 class='main-header'>Neural Data Analysis Pipeline</h1>", unsafe_allow_html=True)
# The line below is modified to set the text color to black
st.markdown("""
<div class='info-box'>
    <span style='color: black;'>
    This interactive application allows you to load, preprocess, and analyze EEG/LFP data, 
    including time-frequency analysis, PAC, and machine learning classification.
    </span>
</div>
""", unsafe_allow_html=True)

# Progress tracking
st.markdown("---")
st.markdown("<h2 class='section-header'>Analysis Progress</h2>", unsafe_allow_html=True)
progress_items = [
    ("Data Loaded", st.session_state.data_loaded),
    ("Preprocessed", st.session_state.data_preprocessed),
    ("Epochs Created", st.session_state.epochs_created),
    ("TFR Computed", st.session_state.tfr_computed),
    ("Features Extracted", st.session_state.features_extracted),
    ("PAC Computed", st.session_state.pac_computed),
    ("ML Completed", st.session_state.ml_completed)
]

for item, status in progress_items:
    if status:
        st.success(f"✅ {item}")
    else:
        st.info(f"⏳ {item}")




st.markdown("---")
st.markdown("<h2 class='section-header'>Results & Visualizations</h2>", unsafe_allow_html=True)

# Power Spectrum Plot
if st.session_state.epochs_created:
    st.markdown("### Power Spectrum")
    if st.session_state.analyzer.epochs and st.session_state.analyzer.epochs.ch_names:
        available_channels_psd = st.session_state.analyzer.epochs.ch_names
        default_channels_psd = available_channels_psd[:min(len(available_channels_psd), 4)]
        
        selected_channels_psd = st.multiselect(
            "Select channels for Power Spectrum:",
            options=available_channels_psd,
            default=default_channels_psd,
            key="psd_channels_multiselect"
        )
        if selected_channels_psd:
            try:
                fig_psd = st.session_state.analyzer.plot_power_spectrum(channels=selected_channels_psd)
                st.pyplot(fig_psd)
            except Exception as e:
                st.error(f"Error plotting power spectrum: {e}")
        else:
            st.info("Select channels to view Power Spectrum.")
    else:
        st.info("No channels available for Power Spectrum plotting.")
else:
    st.info("Create epochs to view Power Spectrum.")


# Time-Frequency Plot
if st.session_state.tfr_computed:
    st.markdown("### Time-Frequency Representation")
    if st.session_state.analyzer.tfr and st.session_state.analyzer.tfr.ch_names:
        available_channels_tfr = st.session_state.analyzer.tfr.ch_names
        selected_channel_tfr = st.selectbox(
            "Select channel for TFR plot:",
            options=available_channels_tfr,
            index=0,
            key="tfr_channel_select"
        )
        
        if selected_channel_tfr:
            try:
                fig_tfr = st.session_state.analyzer.plot_time_frequency(channel=selected_channel_tfr)
                st.pyplot(fig_tfr)
            except Exception as e:
                st.error(f"Error plotting Time-Frequency: {e}")
        else:
            st.info("No channels available for TFR plotting.")
    else:
        st.info("No channels available for TFR plotting.")
else:
    st.info("Compute TFR to view Time-Frequency Representation.")

# PAC Results Plot
if st.session_state.pac_computed:
    st.markdown("### Phase-Amplitude Coupling (PAC) Results")
    if st.session_state.analyzer.pac_results:
        available_channels_pac = list(st.session_state.analyzer.pac_results.keys())
        selected_channel_pac = st.selectbox(
            "Select channel for PAC plot:",
            options=available_channels_pac,
            index=0,
            key="pac_channel_select"
        )
        
        # Add method selection for PAC plotting
        pac_plot_method = st.selectbox(
            "Select PAC plot type:",
            options=['histogram', 'boxplot', 'summary'],
            index=0,
            key="pac_plot_method_select"
        )
        
        if selected_channel_pac:
            try:
                fig_pac = st.session_state.analyzer.plot_pac_results(
                    channel=selected_channel_pac, 
                    method=pac_plot_method
                )
                st.pyplot(fig_pac)
            except Exception as e:
                st.error(f"Error plotting PAC results: {e}")
        else:
            st.info("No PAC results available for plotting.")
    else:
        st.info("No PAC results available for plotting.")
else:
    st.info("Compute PAC to view PAC results.")

# ML Classification Results
if st.session_state.ml_completed:
    st.markdown("### Machine Learning Classification Results")
    if st.session_state.analyzer.ml_results:
        st.success(f"**Classification Accuracy:** {st.session_state.analyzer.ml_results['accuracy']:.2f}")
        st.markdown("#### Classification Report")
        st.code(st.session_state.analyzer.ml_results['classification_report'])
        
        try:
            fig_ml = st.session_state.analyzer.plot_classification_results()
            st.pyplot(fig_ml)
        except Exception as e:
            st.error(f"Error plotting ML results: {e}")
    else:
        st.info("No ML results available.")

else:
    st.info("Run ML classification to view results.")

# Export Results
st.markdown("---")
st.markdown("<h2 class='section-header'>Export Analysis Results</h2>", unsafe_allow_html=True)
if st.session_state.ml_completed: # Assuming results are complete after ML
    export_format = st.radio("Select export format:", ("JSON", "Pickle"), key="export_format_radio")
    
    if st.button("Export Results", key="export_results_btn"):
        with st.spinner("Exporting results..."):
            try:
                output_buffer = io.BytesIO()
                success = st.session_state.analyzer.export_results(output_buffer, format=export_format.lower())
                
                if success:
                    output_buffer.seek(0)
                    if export_format == "JSON":
                        st.download_button(
                            label="Download Results (JSON)",
                            data=output_buffer.getvalue(),
                            file_name="neural_analysis_results.json",
                            mime="application/json"
                        )
                    elif export_format == "Pickle":
                        st.download_button(
                            label="Download Results (Pickle)",
                            data=output_buffer.getvalue(),
                            file_name="neural_analysis_results.pkl",
                            mime="application/octet-stream"
                        )
                    st.success("Results ready for download!")
                else:
                    st.error("Failed to export results.")
            except Exception as e:
                st.error(f"Error during export: {e}")
else:
    st.info("Complete ML Classification to enable results export.")

st.markdown('<div class="end-watermark">Made by Paarth Goela 2023A7PS0006H</div>', unsafe_allow_html=True)

# --- Main Panel: Artifact Visualization ---
if st.session_state.data_loaded:
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Artifact Review</h2>", unsafe_allow_html=True)
    if hasattr(st.session_state.analyzer, 'ica'):
        st.markdown("**ICA Components (Topomap):**")
        try:
            fig = st.session_state.analyzer.ica.plot_components(show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting ICA components: {e}")
    if st.session_state.analyzer.raw is not None:
        st.markdown("**Raw Data (with bad channels marked):**")
        try:
            fig = st.session_state.analyzer.raw.plot(show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting raw data: {e}")
    if st.session_state.epochs_created and hasattr(st.session_state, 'bad_epochs'):
        st.markdown(f"**Bad Epochs Detected:** {st.session_state.bad_epochs}")

# --- ERP Analysis ---
st.sidebar.header("📊 ERP Analysis")
if st.session_state.epochs_created:
    st.sidebar.subheader("Event-Related Potentials")
    if st.sidebar.button("Compute ERPs", key="compute_erps_btn"):
        with st.spinner("Computing ERPs..."):
            try:
                erps = st.session_state.analyzer.compute_erps()
                st.session_state.erps_computed = True
                st.sidebar.success("ERPs computed!")
            except Exception as e:
                st.sidebar.error(f"ERP computation error: {e}")
    if hasattr(st.session_state.analyzer, 'erps'):
        st.sidebar.info("ERPs have been computed. You can plot and compare them.")
        # ERP plotting controls
        st.sidebar.subheader("ERP Visualization")
        # Channel selection for ERP plots
        available_channels = st.session_state.analyzer.epochs.ch_names
        selected_channels_erp = st.sidebar.multiselect(
            "Select channels for ERP plot:",
            options=available_channels,
            default=available_channels[:min(4, len(available_channels))],
            key="erp_channels_select"
        )
        # Condition selection
        available_conditions = list(st.session_state.analyzer.erps.keys())
        selected_conditions_erp = st.sidebar.multiselect(
            "Select conditions for ERP plot:",
            options=available_conditions,
            default=available_conditions,
            key="erp_conditions_select"
        )
        if st.sidebar.button("Plot ERPs", key="plot_erps_btn"):
            try:
                fig = st.session_state.analyzer.plot_erps(
                    channels=selected_channels_erp,
                    conditions=selected_conditions_erp
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting ERPs: {e}")
        # ERP topomap controls
        st.sidebar.subheader("ERP Topomaps")
        time_points_input = st.sidebar.text_input(
            "Time points for topomap (comma-separated, in seconds):",
            value="0.1,0.2,0.3",
            key="erp_time_points_input"
        )
        if st.sidebar.button("Plot ERP Topomaps", key="plot_erp_topomap_btn"):
            try:
                time_points = [float(t.strip()) for t in time_points_input.split(",")]
                fig = st.session_state.analyzer.plot_erp_topomap(
                    time_points=time_points,
                    conditions=selected_conditions_erp
                )
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting ERP topomaps: {e}")
        # Statistical comparison
        st.sidebar.subheader("ERP Statistical Comparison")
        if len(available_conditions) >= 2:
            condition1 = st.sidebar.selectbox(
                "Select first condition:",
                options=available_conditions,
                key="erp_condition1_select"
            )
            condition2 = st.sidebar.selectbox(
                "Select second condition:",
                options=[c for c in available_conditions if c != condition1],
                key="erp_condition2_select"
            )
            alpha = st.sidebar.number_input(
                "Significance level (alpha):",
                value=0.05, min_value=0.001, max_value=0.1, step=0.01, key="erp_alpha_input"
            )
            if st.sidebar.button("Compare ERPs", key="compare_erps_btn"):
                with st.spinner("Comparing ERPs..."):
                    try:
                        results = st.session_state.analyzer.compare_erps(
                            condition1, condition2, alpha=alpha
                        )
                        st.session_state.erp_comparison_results = results
                        st.sidebar.success(f"ERP comparison complete!")
                        # Display results
                        st.markdown("**ERP Comparison Results:**")
                        st.write(f"Method: {results['method']}")
                        st.write(f"Alpha: {results['alpha']}")
                        st.write(f"Significant time points: {np.sum(results['significant'])}")
                        # Plot significant differences
                        if np.sum(results['significant']) > 0:
                            st.markdown("**Significant Differences:**")
                            # Create a simple visualization of significant differences
                            significant_times = np.where(results['significant'].any(axis=0))[0]
                            st.write(f"Significant time indices: {significant_times}")
                    except Exception as e:
                        st.sidebar.error(f"ERP comparison error: {e}")
        else:
            st.sidebar.info("Need at least 2 conditions for comparison.")
else:
    st.sidebar.info("Create epochs first to enable ERP analysis.")

# --- Connectivity Analysis ---
st.sidebar.header("🔗 Connectivity Analysis")
if st.session_state.epochs_created:
    st.sidebar.subheader("Channel Connectivity")
    # Channel pair selection
    available_channels = st.session_state.analyzer.epochs.ch_names
    if len(available_channels) >= 2:
        ch1 = st.sidebar.selectbox(
            "Select first channel:",
            options=available_channels,
            key="connectivity_ch1_select"
        )
        ch2 = st.sidebar.selectbox(
            "Select second channel:",
            options=[c for c in available_channels if c != ch1],
            key="connectivity_ch2_select"
        )
        # Connectivity method selection
        connectivity_method = st.sidebar.selectbox(
            "Connectivity method:",
            options=['coherence', 'plv', 'granger'],
            key="connectivity_method_select"
        )
        if st.sidebar.button("Compute Connectivity", key="compute_connectivity_btn"):
            with st.spinner("Computing connectivity..."):
                try:
                    if connectivity_method == 'coherence':
                        freqs, coh = st.session_state.analyzer.compute_coherence(ch1, ch2)
                        st.session_state.connectivity_results = {
                            'method': 'coherence',
                            'freqs': freqs,
                            'values': coh,
                            'ch1': ch1,
                            'ch2': ch2
                        }
                        st.sidebar.success("Coherence computed!")
                    elif connectivity_method == 'plv':
                        freqs, plv = st.session_state.analyzer.compute_plv(ch1, ch2)
                        st.session_state.connectivity_results = {
                            'method': 'plv',
                            'freqs': freqs,
                            'values': plv,
                            'ch1': ch1,
                            'ch2': ch2
                        }
                        st.sidebar.success("PLV computed!")
                    elif connectivity_method == 'granger':
                        results = st.session_state.analyzer.compute_granger_causality(ch1, ch2)
                        st.session_state.connectivity_results = {
                            'method': 'granger',
                            'results': results,
                            'ch1': ch1,
                            'ch2': ch2
                        }
                        st.sidebar.success("Granger causality computed!")
                except Exception as e:
                    st.sidebar.error(f"Connectivity computation error: {e}")
        # Display connectivity results
        if hasattr(st.session_state, 'connectivity_results'):
            results = st.session_state.connectivity_results
            st.markdown("**Connectivity Results:**")
            st.write(f"Method: {results['method']}")
            st.write(f"Channels: {results['ch1']} → {results['ch2']}")
            if results['method'] in ['coherence', 'plv']:
                # Plot connectivity spectrum
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results['freqs'], results['values'], linewidth=2)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel(f'{results["method"].upper()}')
                ax.set_title(f'{results["method"].upper()} between {results["ch1"]} and {results["ch2"]}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            elif results['method'] == 'granger':
                granger_results = results['results']
                st.write(f"F-statistic: {granger_results['f_statistic']:.4f}")
                st.write(f"P-value: {granger_results['p_value']:.4f}")
                st.write(f"Significant: {granger_results['significant']}")
    # Connectivity matrix
    st.sidebar.subheader("Connectivity Matrix")
    matrix_method = st.sidebar.selectbox(
        "Matrix method:",
        options=['coherence', 'plv'],
        key="matrix_method_select"
    )
    if st.sidebar.button("Plot Connectivity Matrix", key="plot_connectivity_matrix_btn"):
        with st.spinner("Computing connectivity matrix..."):
            try:
                fig = st.session_state.analyzer.plot_connectivity_matrix(method=matrix_method)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error plotting connectivity matrix: {e}")
else:
    st.sidebar.info("Create epochs first to enable connectivity analysis.")

# --- Source Localization ---
st.sidebar.header("🧠 Source Localization")
if st.session_state.data_loaded:
    st.sidebar.subheader("Source Space Setup")
    if st.sidebar.button("Setup Source Space", key="setup_source_space_btn"):
        with st.spinner("Setting up source space..."):
            try:
                src = st.session_state.analyzer.setup_source_space()
                if src is not None:
                    st.session_state.source_space_setup = True
                    st.sidebar.success("Source space setup complete!")
                else:
                    st.sidebar.warning("Source space setup failed. Check data format.")
            except Exception as e:
                st.sidebar.error(f"Source space setup error: {e}")
    
    if hasattr(st.session_state.analyzer, 'src') and st.session_state.analyzer.src is not None:
        st.sidebar.info("Source space is set up. You can compute forward model.")
        st.sidebar.subheader("Forward Model")
        if st.sidebar.button("Compute Forward Model", key="compute_forward_btn"):
            with st.spinner("Computing forward model..."):
                try:
                    fwd = st.session_state.analyzer.compute_forward_model()
                    if fwd is not None:
                        st.session_state.forward_computed = True
                        st.sidebar.success("Forward model computed!")
                    else:
                        st.sidebar.warning("Forward model computation failed.")
                except Exception as e:
                    st.sidebar.error(f"Forward model error: {e}")
        
        if hasattr(st.session_state.analyzer, 'fwd') and st.session_state.analyzer.fwd is not None:
            st.sidebar.info("Forward model is computed. You can compute inverse solution.")
            st.sidebar.subheader("Inverse Solution")
            inverse_method = st.sidebar.selectbox(
                "Inverse method:",
                options=['dSPM', 'LORETA'],
                key="inverse_method_select"
            )
            lambda2 = st.sidebar.number_input(
                "Lambda2 (regularization):",
                value=1.0/9.0, min_value=0.001, max_value=1.0, step=0.01,
                key="lambda2_input"
            )
            if st.sidebar.button("Compute Inverse Solution", key="compute_inverse_btn"):
                with st.spinner("Computing inverse solution..."):
                    try:
                        stc = st.session_state.analyzer.compute_inverse_solution(
                            method=inverse_method, lambda2=lambda2
                        )
                        if stc is not None:
                            st.session_state.inverse_computed = True
                            st.sidebar.success(f"Inverse solution computed using {inverse_method}!")
                        else:
                            st.sidebar.warning("Inverse solution computation failed.")
                    except Exception as e:
                        st.sidebar.error(f"Inverse solution error: {e}")
            
            if hasattr(st.session_state.analyzer, 'stc') and st.session_state.analyzer.stc is not None:
                st.sidebar.info("Inverse solution is computed. You can visualize sources.")
                st.sidebar.subheader("Source Visualization")
                views = st.sidebar.multiselect(
                    "Select views:",
                    options=['lat', 'med', 'ros', 'cau'],
                    default=['lat', 'med'],
                    key="source_views_select"
                )
                if st.sidebar.button("Plot Source Estimates", key="plot_sources_btn"):
                    try:
                        fig = st.session_state.analyzer.plot_source_estimates(views=views)
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error plotting source estimates: {e}")
else:
    st.sidebar.info("Load data first to enable source localization.")

# --- Batch Processing & Reporting ---
st.sidebar.header("📊 Batch Processing & Reporting")

# Batch file upload
st.sidebar.subheader("Batch File Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload multiple files for batch processing:",
    type=["edf", "fif", "set"],
    accept_multiple_files=True,
    key="batch_files_uploader"
)

if uploaded_files:
    st.sidebar.info(f"Uploaded {len(uploaded_files)} files for batch processing.")
    
    # Batch processing parameters
    st.sidebar.subheader("Batch Processing Parameters")
    
    # Preprocessing parameters
    with st.sidebar.expander("Preprocessing Parameters"):
        l_freq = st.number_input("Low frequency (Hz)", value=1.0, key="batch_l_freq")
        h_freq = st.number_input("High frequency (Hz)", value=40.0, key="batch_h_freq")
        notch_freq = st.number_input("Notch frequency (Hz)", value=50.0, key="batch_notch_freq")
    
    # Analysis parameters
    with st.sidebar.expander("Analysis Parameters"):
        tmin = st.number_input("Epoch start (s)", value=-0.5, key="batch_tmin")
        tmax = st.number_input("Epoch end (s)", value=1.0, key="batch_tmax")
        pac_phase_min = st.number_input("PAC phase min (Hz)", value=4.0, key="batch_pac_phase_min")
        pac_phase_max = st.number_input("PAC phase max (Hz)", value=8.0, key="batch_pac_phase_max")
        pac_amp_min = st.number_input("PAC amplitude min (Hz)", value=30.0, key="batch_pac_amp_min")
        pac_amp_max = st.number_input("PAC amplitude max (Hz)", value=40.0, key="batch_pac_amp_max")
    
    if st.sidebar.button("Start Batch Processing", key="start_batch_btn"):
        with st.spinner("Processing files in batch..."):
            try:
                # Save uploaded files temporarily
                import tempfile
                import os
                temp_files = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_files.append(tmp_file.name)
                
                # Prepare parameters
                preprocessing_params = {
                    'l_freq': l_freq,
                    'h_freq': h_freq,
                    'notch_freq': notch_freq
                }
                
                analysis_params = {
                    'epoch_params': {'tmin': tmin, 'tmax': tmax},
                    'pac_params': {
                        'low_freq': (pac_phase_min, pac_phase_max),
                        'high_freq': (pac_amp_min, pac_amp_max)
                    }
                }
                
                # Run batch processing
                results, batch_dir = st.session_state.analyzer.batch_process_files(
                    temp_files, 
                    preprocessing_params=preprocessing_params,
                    analysis_params=analysis_params
                )
                
                st.session_state.batch_results = results
                st.session_state.batch_dir = batch_dir
                
                # Clean up temp files
                for temp_file in temp_files:
                    os.unlink(temp_file)
                
                st.sidebar.success(f"Batch processing complete! Results saved to {batch_dir}")
                
            except Exception as e:
                st.sidebar.error(f"Batch processing error: {e}")

# Display batch results
if hasattr(st.session_state, 'batch_results'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Batch Processing Results</h2>", unsafe_allow_html=True)
    
    # Create results table
    results_data = []
    for filename, result in st.session_state.batch_results.items():
        if 'error' not in result:
            results_data.append({
                'Filename': filename,
                'Epochs': result.get('n_epochs', 'N/A'),
                'Channels': result.get('n_channels', 'N/A'),
                'ERPs': 'Yes' if result.get('erps', False) else 'No',
                'ML Accuracy': f"{result.get('ml_accuracy', 'N/A'):.3f}" if result.get('ml_accuracy') is not None else 'N/A'
            })
        else:
            results_data.append({
                'Filename': filename,
                'Epochs': 'Error',
                'Channels': 'Error',
                'ERPs': 'Error',
                'ML Accuracy': 'Error'
            })
    
    if results_data:
        import pandas as pd
        df = pd.DataFrame(results_data)
        st.dataframe(df)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Batch Results (CSV)",
            data=csv,
            file_name="batch_results.csv",
            mime="text/csv"
        )

# Report generation
st.sidebar.subheader("Report Generation")
if st.sidebar.button("Generate Analysis Report", key="generate_report_btn"):
    with st.spinner("Generating report..."):
        try:
            report_path = st.session_state.analyzer.generate_report()
            st.sidebar.success(f"Report generated: {report_path}")
            
            # Display report
            with open(report_path, 'r') as f:
                report_content = f.read()
            st.markdown("---")
            st.markdown("<h2 class='section-header'>Analysis Report</h2>", unsafe_allow_html=True)
            st.components.v1.html(report_content, height=600)
            
        except Exception as e:
            st.sidebar.error(f"Report generation error: {e}")

# --- STATISTICAL ANALYSIS ---
st.sidebar.header("📊 Statistical Analysis")

if st.session_state.epochs_created:
    st.sidebar.subheader("Statistical Tests")
    
    # Statistical test parameters
    test_type = st.sidebar.selectbox(
        "Test type:",
        options=['parametric', 'nonparametric'],
        key="stat_test_type"
    )
    
    alpha = st.sidebar.number_input(
        "Significance level (α):",
        value=0.05, min_value=0.001, max_value=0.1, step=0.01,
        key="stat_alpha"
    )
    
    correction = st.sidebar.selectbox(
        "Multiple comparison correction:",
        options=['fdr_bh', 'bonferroni', 'holm'],
        key="stat_correction"
    )
    
    if st.sidebar.button("Run Statistical Analysis", key="run_stats_btn"):
        with st.spinner("Performing statistical analysis..."):
            try:
                stats_results = st.session_state.analyzer.statistical_analysis(
                    test_type=test_type,
                    alpha=alpha,
                    correction=correction
                )
                st.session_state.stats_results = stats_results
                st.sidebar.success("Statistical analysis completed!")
                
            except Exception as e:
                st.sidebar.error(f"Statistical analysis error: {e}")
    
    # Cluster permutation test
    if st.session_state.tfr_computed:
        st.sidebar.subheader("Cluster Permutation Test")
        n_permutations = st.sidebar.number_input(
            "Number of permutations:",
            value=1000, min_value=100, max_value=10000, step=100,
            key="n_permutations"
        )
        
        if st.sidebar.button("Run Cluster Test", key="run_cluster_btn"):
            with st.spinner("Running cluster permutation test..."):
                try:
                    cluster_results = st.session_state.analyzer.cluster_permutation_test(
                        n_permutations=n_permutations,
                        alpha=alpha
                    )
                    st.session_state.cluster_results = cluster_results
                    st.sidebar.success("Cluster test completed!")
                    
                except Exception as e:
                    st.sidebar.error(f"Cluster test error: {e}")

# --- CUSTOM FREQUENCY BANDS ---
st.sidebar.header("🎵 Custom Frequency Bands")

# Define custom bands
st.sidebar.subheader("Define Frequency Bands")
custom_bands_input = st.sidebar.text_area(
    "Enter custom frequency bands (JSON format):",
    value='{"alpha": [8, 13], "beta": [13, 30], "gamma": [30, 80], "theta": [4, 8]}',
    key="custom_bands_input"
)

if st.sidebar.button("Define Custom Bands", key="define_bands_btn"):
    try:
        import json
        bands_dict = json.loads(custom_bands_input)
        # Convert lists to tuples
        bands_dict = {k: tuple(v) for k, v in bands_dict.items()}
        st.session_state.analyzer.define_custom_frequency_bands(bands_dict)
        st.sidebar.success("Custom frequency bands defined!")
    except Exception as e:
        st.sidebar.error(f"Error defining bands: {e}")

# Analyze custom bands
if hasattr(st.session_state.analyzer, 'custom_bands') and st.session_state.epochs_created:
    st.sidebar.subheader("Analyze Custom Bands")
    
    analysis_method = st.sidebar.selectbox(
        "Analysis method:",
        options=['power', 'phase', 'both'],
        key="custom_analysis_method"
    )
    
    if st.sidebar.button("Analyze Custom Bands", key="analyze_custom_btn"):
        with st.spinner("Analyzing custom frequency bands..."):
            try:
                custom_results = st.session_state.analyzer.analyze_custom_bands(
                    method=analysis_method
                )
                st.session_state.custom_band_results = custom_results
                st.sidebar.success("Custom band analysis completed!")
                
            except Exception as e:
                st.sidebar.error(f"Custom band analysis error: {e}")

# --- BEHAVIORAL DATA INTEGRATION ---
st.sidebar.header("🧠 Behavioral Data Integration")

# Load behavioral data
st.sidebar.subheader("Load Behavioral Data")
behavioral_file = st.sidebar.file_uploader(
    "Upload behavioral data file:",
    type=["csv", "xlsx", "json"],
    key="behavioral_file_uploader"
)

if behavioral_file is not None:
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(behavioral_file.name)[1]) as tmp_file:
            tmp_file.write(behavioral_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load behavioral data
        data_format = os.path.splitext(behavioral_file.name)[1][1:]  # Remove the dot
        if data_format == 'xlsx':
            data_format = 'excel'
        
        success = st.session_state.analyzer.load_behavioral_data(tmp_path, data_format)
        if success:
            st.sidebar.success("Behavioral data loaded!")
            st.session_state.behavioral_loaded = True
            
            # Show behavioral data columns
            if hasattr(st.session_state.analyzer, 'behavioral_data'):
                st.sidebar.info(f"Columns: {', '.join(st.session_state.analyzer.behavioral_data.columns)}")
        
        # Clean up temp file
        os.unlink(tmp_path)
        
    except Exception as e:
        st.sidebar.error(f"Error loading behavioral data: {e}")

# Neural-behavioral correlation
if hasattr(st.session_state, 'behavioral_loaded') and st.session_state.behavioral_loaded and st.session_state.epochs_created:
    st.sidebar.subheader("Neural-Behavioral Correlation")
    
    neural_measure = st.sidebar.selectbox(
        "Neural measure:",
        options=['power', 'erp'],
        key="neural_measure_select"
    )
    
    if hasattr(st.session_state.analyzer, 'behavioral_data'):
        behavioral_column = st.sidebar.selectbox(
            "Behavioral column:",
            options=st.session_state.analyzer.behavioral_data.columns.tolist(),
            key="behavioral_column_select"
        )
        
        if st.sidebar.button("Compute Correlation", key="compute_correlation_btn"):
            with st.spinner("Computing neural-behavioral correlation..."):
                try:
                    correlations = st.session_state.analyzer.correlate_neural_behavioral(
                        neural_measure=neural_measure,
                        behavioral_column=behavioral_column
                    )
                    st.session_state.neural_behavioral_correlations = correlations
                    st.sidebar.success("Correlation analysis completed!")
                    
                except Exception as e:
                    st.sidebar.error(f"Correlation analysis error: {e}")

# --- ENHANCED EXPORT & REPRODUCIBILITY ---
st.sidebar.header("📤 Export & Reproducibility")

# Export analysis pipeline
st.sidebar.subheader("Export Analysis Pipeline")
if st.sidebar.button("Export Pipeline", key="export_pipeline_btn"):
    with st.spinner("Exporting analysis pipeline..."):
        try:
            export_dir = st.session_state.analyzer.export_analysis_pipeline()
            st.sidebar.success(f"Pipeline exported to: {export_dir}")
            
            # Create download link for the exported directory
            import zipfile
            import io
            
            # Create a zip file of the exported directory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(export_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, export_dir)
                        zip_file.write(file_path, arcname)
            
            zip_buffer.seek(0)
            st.sidebar.download_button(
                label="Download Exported Pipeline (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="neural_analysis_pipeline.zip",
                mime="application/zip"
            )
            
        except Exception as e:
            st.sidebar.error(f"Export error: {e}")

# Export to BIDS format
st.sidebar.subheader("Export to BIDS Format")
if st.sidebar.button("Export to BIDS", key="export_bids_btn"):
    with st.spinner("Exporting to BIDS format..."):
        try:
            bids_dir = st.session_state.analyzer.export_to_bids()
            st.sidebar.success(f"BIDS export completed: {bids_dir}")
            
        except Exception as e:
            st.sidebar.error(f"BIDS export error: {e}")

# --- INTERACTIVE EXPLORATION ---
st.sidebar.header("🔍 Interactive Exploration")

# Interactive plots
st.sidebar.subheader("Interactive Plots")
plot_type = st.sidebar.selectbox(
    "Plot type:",
    options=['time_frequency', 'erp', 'connectivity'],
    key="interactive_plot_type"
)

if st.sidebar.button("Create Interactive Plot", key="create_interactive_btn"):
    with st.spinner("Creating interactive plot..."):
        try:
            interactive_fig = st.session_state.analyzer.create_interactive_plot(plot_type, theme=st.session_state.theme)
            if interactive_fig is not None:
                st.session_state.interactive_fig = interactive_fig
                st.sidebar.success("Interactive plot created!")
            else:
                st.sidebar.warning("Could not create interactive plot")
                
        except Exception as e:
            st.sidebar.error(f"Interactive plot error: {e}")

# Dashboard data
if st.sidebar.button("Generate Dashboard Data", key="generate_dashboard_btn"):
    with st.spinner("Generating dashboard data..."):
        try:
            dashboard_data = st.session_state.analyzer.create_dashboard_data()
            st.session_state.dashboard_data = dashboard_data
            st.sidebar.success("Dashboard data generated!")
            
        except Exception as e:
            st.sidebar.error(f"Dashboard generation error: {e}")

# --- MAIN CONTENT AREA ---
st.markdown("<h1 class='main-header'>🧠 Neural Data Analysis Pipeline</h1>", unsafe_allow_html=True)

# Route to different features based on selection
if advanced_feature == "Real-Time EEG":
    if REALTIME_AVAILABLE:
        create_realtime_dashboard()
    else:
        st.error("Real-time EEG module not available. Please check the installation.")
        
elif advanced_feature == "Advanced Visualization":
    if VISUALIZATION_AVAILABLE:
        create_advanced_visualization_interface()
    else:
        st.error("Advanced Visualization module not available. Please check the installation.")
        
elif advanced_feature == "Advanced Analysis":
    if ANALYSIS_AVAILABLE:
        create_advanced_analysis_interface()
    else:
        st.error("Advanced Analysis module not available. Please check the installation.")
        
elif advanced_feature == "Data Pipeline":
    if PIPELINE_AVAILABLE:
        st.markdown("### 📊 Data Pipeline Dashboard")
        
        # Display pipeline results
        if hasattr(st.session_state, 'pipeline_results'):
            st.markdown("---")
            st.markdown("<h2 class='section-header'>Pipeline Results</h2>", unsafe_allow_html=True)
            
            results = st.session_state.pipeline_results
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", len(results))
            with col2:
                st.metric("Successful", successful)
            with col3:
                st.metric("Failed", failed)
            
            # Create results table
            results_data = []
            for result in results:
                results_data.append({
                    'File': os.path.basename(result.file_path),
                    'Status': '✅ Success' if result.success else '❌ Failed',
                    'Processing Time': f"{result.processing_time:.2f}s",
                    'Output Files': len(result.output_files),
                    'Errors': '; '.join(result.errors) if result.errors else 'None'
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Pipeline Results (CSV)",
                data=csv,
                file_name="pipeline_results.csv",
                mime="text/csv"
            )
        
        # Display validation results
        if hasattr(st.session_state, 'validation_results'):
            st.markdown("---")
            st.markdown("<h2 class='section-header'>Data Validation Results</h2>", unsafe_allow_html=True)
            
            validation = st.session_state.validation_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", validation['total_files'])
            with col2:
                st.metric("Valid Files", validation['valid_files'])
            with col3:
                st.metric("Invalid Files", validation['invalid_files'])
            
            if validation['duplicates']:
                st.warning(f"Found {len(validation['duplicates'])} duplicate files")
                for dup in validation['duplicates']:
                    st.write(f"Duplicate: {dup['file1']} and {dup['file2']}")
            
            if validation['errors']:
                st.error("Validation errors found:")
                for error in validation['errors']:
                    st.write(f"- {error}")
    else:
        st.error("Data Pipeline module not available. Please check the installation.")
        
elif advanced_feature == "Workflow Automation":
    if WORKFLOW_AVAILABLE:
        st.markdown("### 🔄 Workflow Automation Dashboard")
        
        # Display workflow results
        if hasattr(st.session_state, 'workflow_result'):
            st.markdown("---")
            st.markdown("<h2 class='section-header'>Workflow Execution Results</h2>", unsafe_allow_html=True)
            
            workflow_result = st.session_state.workflow_result
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Steps Executed", workflow_result['steps_executed'])
            with col2:
                st.metric("Steps Successful", workflow_result['steps_successful'])
            with col3:
                st.metric("Steps Failed", workflow_result['steps_failed'])
            with col4:
                st.metric("Total Time", f"{workflow_result['total_execution_time']:.2f}s")
            
            # Display step details
            if workflow_result['step_results']:
                st.subheader("Step Details")
                step_data = []
                for step_name, step_result in workflow_result['step_results'].items():
                    step_data.append({
                        'Step': step_name,
                        'Status': '✅ Success' if step_result['success'] else '❌ Failed',
                        'Execution Time': f"{step_result['execution_time']:.2f}s",
                        'Cached': 'Yes' if step_result.get('cached', False) else 'No',
                        'Error': step_result.get('error', 'None')
                    })
                
                df = pd.DataFrame(step_data)
                st.dataframe(df)
        
        # Show workflow help
        st.markdown("---")
        st.markdown("<h2 class='section-header'>Workflow Management</h2>", unsafe_allow_html=True)
        
        st.info("""
        **Workflow Automation Features:**
        - Create reproducible analysis workflows
        - Execute workflows with dependency tracking
        - Cache intermediate results for efficiency
        - Track execution logs and performance
        - Export workflows for sharing and reproducibility
        """)
    else:
        st.error("Workflow Automation module not available. Please check the installation.")
        
elif advanced_feature == "Data Management":
    if DATA_MANAGEMENT_AVAILABLE:
        st.markdown("### 🗂️ Data Management Dashboard")
        
        # BIDS validation
        if hasattr(st.session_state, 'bids_manager'):
            st.markdown("---")
            st.markdown("<h2 class='section-header'>BIDS Dataset</h2>", unsafe_allow_html=True)
            
            if st.button("Validate BIDS Structure", key="validate_bids_btn"):
                with st.spinner("Validating BIDS structure..."):
                    try:
                        validation = st.session_state.bids_manager.validate_bids_structure()
                        st.session_state.bids_validation = validation
                        st.success("BIDS validation completed!")
                        
                    except Exception as e:
                        st.error(f"BIDS validation error: {e}")
            
            if hasattr(st.session_state, 'bids_validation'):
                validation = st.session_state.bids_validation
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Valid", "✅" if validation['valid'] else "❌")
                with col2:
                    st.metric("Total Files", validation['total_files'])
                with col3:
                    st.metric("Participants", len(validation['participants']))
                
                if validation['errors']:
                    st.error("BIDS validation errors:")
                    for error in validation['errors']:
                        st.write(f"- {error}")
                
                if validation['warnings']:
                    st.warning("BIDS validation warnings:")
                    for warning in validation['warnings']:
                        st.write(f"- {warning}")
        
        # Database summary
        if hasattr(st.session_state, 'db_manager'):
            st.markdown("---")
            st.markdown("<h2 class='section-header'>Database Summary</h2>", unsafe_allow_html=True)
            
            if st.button("Get Database Summary", key="db_summary_btn"):
                try:
                    summary = st.session_state.db_manager.get_database_summary()
                    st.session_state.db_summary = summary
                    
                except Exception as e:
                    st.error(f"Database summary error: {e}")
            
            if hasattr(st.session_state, 'db_summary'):
                summary = st.session_state.db_summary
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files", summary['total_files'])
                with col2:
                    st.metric("Participants", summary['total_participants'])
                with col3:
                    st.metric("Analyses", summary['total_analyses'])
                with col4:
                    st.metric("Workflows", summary['total_workflows'])
                
                # File types distribution
                if summary['file_types']:
                    st.subheader("File Types")
                    file_types_df = pd.DataFrame([
                        {'Type': k, 'Count': v} for k, v in summary['file_types'].items()
                    ])
                    st.dataframe(file_types_df)
        
        # Data organization summary
        st.markdown("---")
        st.markdown("<h2 class='section-header'>Data Organization</h2>", unsafe_allow_html=True)
        
        if st.button("Get Organization Summary", key="org_summary_btn"):
            try:
                organizer = DataOrganizer()
                summary = organizer.get_organization_summary()
                st.session_state.org_summary = summary
                
            except Exception as e:
                st.error(f"Organization summary error: {e}")
        
        if hasattr(st.session_state, 'org_summary'):
            summary = st.session_state.org_summary
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Files", summary['total_files'])
            with col2:
                st.metric("Total Size", f"{summary['total_size'] / (1024*1024):.1f} MB")
            with col3:
                st.metric("File Types", len(summary['file_types']))
            
            if summary['participants']:
                st.subheader("Participants")
                st.write(f"Found {len(summary['participants'])} participants: {', '.join(summary['participants'])}")
            
            if summary['dates']:
                st.subheader("Date Range")
                dates = sorted(summary['dates'])
                st.write(f"From {dates[0]} to {dates[-1]}")
    else:
        st.error("Data Management module not available. Please check the installation.")
        
elif advanced_feature == "Enhanced UX":
    if UX_AVAILABLE:
        st.markdown("### 🎨 Enhanced UX Dashboard")
        
        # Show keyboard shortcuts help
        if hasattr(st.session_state, 'show_shortcuts_help') and st.session_state.show_shortcuts_help:
            render_keyboard_shortcuts_help()
            if st.button("Close Help", key="close_shortcuts_help_btn"):
                st.session_state.show_shortcuts_help = False
                st.rerun()
        
        # Show dashboard customization
        if hasattr(st.session_state, 'show_dashboard_customization') and st.session_state.show_dashboard_customization:
            render_dashboard_customization()
            if st.button("Close Customization", key="close_dashboard_customization_btn"):
                st.session_state.show_dashboard_customization = False
                st.rerun()
        
        # Performance and accessibility info
        st.markdown("---")
        st.markdown("<h2 class='section-header'>Enhanced Features</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Performance Features")
            st.info("""
            - **Performance Mode**: Optimize for speed
            - **Caching**: Smart result caching
            - **Auto-refresh**: Real-time updates
            - **Parallel Processing**: Multi-core utilization
            """)
        
        with col2:
            st.subheader("♿ Accessibility Features")
            st.info("""
            - **High Contrast**: Better visibility
            - **Large Fonts**: Improved readability
            - **Keyboard Navigation**: Full keyboard support
            - **Screen Reader**: Accessibility compliance
            """)
        
        st.subheader("⌨️ Keyboard Shortcuts")
        st.info("""
        **Quick Actions:**
        - `Ctrl+Shift+U`: Upload Data
        - `Ctrl+Shift+P`: Preprocess Data
        - `Ctrl+Shift+A`: Run Analysis
        - `Ctrl+Shift+V`: Advanced Visualization
        - `Ctrl+Shift+R`: Real-time EEG
        - `Ctrl+Shift+W`: Workflow Manager
        - `Ctrl+Shift+D`: Toggle Dark Mode
        - `Ctrl+Shift+S`: Save Session
        - `Ctrl+Shift+L`: Load Session
        - `Ctrl+Shift+E`: Export Results
        - `Ctrl+Shift+H`: Show Help
        - `Ctrl+Shift+C`: Clear All
        
        **Function Keys:**
        - `F1`: Quick Help
        - `F2`: Toggle Sidebar
        - `F3`: Fullscreen Mode
        - `F4`: Reset View
        - `F5`: Refresh Data
        - `F6`: Next Analysis
        - `F7`: Previous Analysis
        - `F8`: Toggle Auto-save
        - `F9`: Performance Mode
        - `F10`: Debug Mode
        - `F11`: Screenshot
        - `F12`: Developer Tools
        """)
    else:
        st.error("Enhanced UX module not available. Please check the installation.")
        
else:
    # Main analysis content
    st.markdown("### 📊 Main Analysis Dashboard")

# Display statistical analysis results
if hasattr(st.session_state, 'stats_results'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Statistical Analysis Results</h2>", unsafe_allow_html=True)
    
    stats = st.session_state.stats_results
    summary = stats.get('summary', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Channels", summary.get('total_channels', 'N/A'))
    with col2:
        st.metric("Significant Channels", summary.get('significant_channels', 'N/A'))
    with col3:
        st.metric("Significant %", f"{summary.get('significant_percentage', 0):.1f}%")
    
    # Display channel-wise results
    if 'channel_stats' in stats:
        channel_data = []
        for ch_name, ch_stats in stats['channel_stats'].items():
            channel_data.append({
                'Channel': ch_name,
                'Test': ch_stats.get('test', 'N/A'),
                'Statistic': f"{ch_stats.get('statistic', 0):.3f}",
                'P-value': f"{ch_stats.get('p_value', 1):.3f}",
                'Significant': 'Yes' if ch_stats.get('significant', False) else 'No',
                'Corrected P-value': f"{ch_stats.get('p_corrected', 1):.3f}",
                'Significant (Corrected)': 'Yes' if ch_stats.get('significant_corrected', False) else 'No'
            })
        
        df = pd.DataFrame(channel_data)
        st.dataframe(df)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Statistical Results (CSV)",
            data=csv,
            file_name="statistical_results.csv",
            mime="text/csv"
        )

# Display cluster permutation test results
if hasattr(st.session_state, 'cluster_results'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Cluster Permutation Test Results</h2>", unsafe_allow_html=True)
    
    cluster_results = st.session_state.cluster_results
    significant_clusters = cluster_results.get('significant_clusters', [])
    
    st.metric("Significant Clusters", len(significant_clusters))
    
    if significant_clusters:
        cluster_data = []
        for cluster in significant_clusters:
            cluster_data.append({
                'Cluster ID': cluster['cluster_id'],
                'T-statistic': f"{cluster['t_stat']:.3f}",
                'P-value': f"{cluster['p_value']:.3f}",
                'Size': cluster['size']
            })
        
        df = pd.DataFrame(cluster_data)
        st.dataframe(df)

# Display custom frequency band results
if hasattr(st.session_state, 'custom_band_results'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Custom Frequency Band Analysis</h2>", unsafe_allow_html=True)
    
    custom_results = st.session_state.custom_band_results
    
    # Create tabs for different bands
    band_names = list(custom_results.keys())
    if band_names:
        tabs = st.tabs(band_names)
        
        for i, (band_name, results) in enumerate(custom_results.items()):
            with tabs[i]:
                if 'power' in results:
                    st.subheader(f"{band_name} Band Power")
                    
                    # Create power plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    mean_power = results['mean_power']
                    std_power = results['std_power']
                    
                    ax.bar(range(len(mean_power)), mean_power, yerr=std_power)
                    ax.set_title(f'{band_name} Band Power by Channel')
                    ax.set_xlabel('Channel Index')
                    ax.set_ylabel('Power')
                    ax.tick_params(axis='x', rotation=45)
                    
                    st.pyplot(fig)
                    
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Power", f"{np.mean(mean_power):.3f}")
                    with col2:
                        st.metric("Max Power", f"{np.max(mean_power):.3f}")
                    with col3:
                        st.metric("Min Power", f"{np.min(mean_power):.3f}")

# Display neural-behavioral correlation results
if hasattr(st.session_state, 'neural_behavioral_correlations'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Neural-Behavioral Correlations</h2>", unsafe_allow_html=True)
    
    correlations = st.session_state.neural_behavioral_correlations
    
    # Create correlation heatmap
    corr_values = [stats['correlation'] for stats in correlations.values()]
    p_values = [stats['p_value'] for stats in correlations.values()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(np.array(corr_values).reshape(1, -1), cmap='RdBu_r', 
                   aspect='auto', vmin=-1, vmax=1)
    
    # Mark significant correlations
    sig_mask = np.array(p_values) < 0.05
    for i, sig in enumerate(sig_mask):
        if sig:
            ax.text(i, 0, '*', ha='center', va='center', color='black', fontsize=12)
    
    ax.set_xticks(range(len(correlations)))
    ax.set_xticklabels(list(correlations.keys()), rotation=45)
    ax.set_yticks([])
    ax.set_title('Neural-Behavioral Correlations')
    plt.colorbar(im, ax=ax)
    
    st.pyplot(fig)
    
    # Display correlation table
    corr_data = []
    for ch_name, stats in correlations.items():
        corr_data.append({
            'Channel': ch_name,
            'Correlation': f"{stats['correlation']:.3f}",
            'P-value': f"{stats['p_value']:.3f}",
            'Significant': 'Yes' if stats['p_value'] < 0.05 else 'No'
        })
    
    df = pd.DataFrame(corr_data)
    st.dataframe(df)
    
    # Download results
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Correlation Results (CSV)",
        data=csv,
        file_name="neural_behavioral_correlations.csv",
        mime="text/csv"
    )

# Display interactive plots
if hasattr(st.session_state, 'interactive_fig'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Interactive Plot</h2>", unsafe_allow_html=True)
    st.plotly_chart(st.session_state.interactive_fig, use_container_width=True)

# Display dashboard data
if hasattr(st.session_state, 'dashboard_data'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    dashboard = st.session_state.dashboard_data
    
    # Create metrics display
    if 'epochs' in dashboard:
        st.subheader("Data Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Epochs", dashboard['epochs']['n_epochs'])
        with col2:
            st.metric("Channels", dashboard['epochs']['n_channels'])
        with col3:
            st.metric("Duration (s)", f"{dashboard['epochs']['duration']:.2f}")
        with col4:
            st.metric("Sampling Rate (Hz)", f"{dashboard['epochs']['sampling_rate']:.0f}")
    
    if 'ml' in dashboard:
        st.subheader("Machine Learning Results")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{dashboard['ml']['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{dashboard['ml']['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{dashboard['ml']['recall']:.3f}")
        with col4:
            st.metric("F1 Score", f"{dashboard['ml']['f1_score']:.3f}")
    
    if 'pac' in dashboard:
        st.subheader("PAC Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean PAC", f"{dashboard['pac']['mean_pac']:.3f}")
        with col2:
            st.metric("Max PAC", f"{dashboard['pac']['max_pac']:.3f}")
        with col3:
            st.metric("Channels with PAC", dashboard['pac']['n_channels'])

# --- ADVANCED FEATURES ---
st.sidebar.header("🚀 Advanced Features")

# Advanced feature selection
advanced_feature = st.sidebar.selectbox(
    "Select Advanced Feature:",
    options=[
        "Main Analysis",
        "Real-Time EEG", 
        "Advanced Visualization",
        "Advanced Analysis",
        "Data Pipeline",
        "Workflow Automation", 
        "Data Management",
        "Enhanced UX"
    ],
    key="advanced_feature_select"
)

# --- SIGNAL QUALITY METRICS ---
st.sidebar.subheader("📊 Signal Quality Metrics")

if st.session_state.epochs_created:
    quality_metrics_list = st.sidebar.multiselect(
        "Select metrics to compute:",
        options=['variance', 'entropy', 'kurtosis', 'snr', 'line_noise'],
        default=['variance', 'snr'],
        key="quality_metrics_select"
    )
    
    if st.sidebar.button("Compute Quality Metrics", key="compute_quality_btn"):
        with st.spinner("Computing signal quality metrics..."):
            try:
                quality_metrics = st.session_state.analyzer.compute_signal_quality_metrics(
                    metrics=quality_metrics_list
                )
                st.session_state.quality_metrics = quality_metrics
                st.sidebar.success("Quality metrics computed!")
                
            except Exception as e:
                st.sidebar.error(f"Quality metrics error: {e}")
    
    # Plot quality metrics
    if hasattr(st.session_state, 'quality_metrics'):
        metric_to_plot = st.sidebar.selectbox(
            "Plot metric:",
            options=quality_metrics_list,
            key="quality_metric_plot"
        )
        
        if st.sidebar.button("Plot Quality Metrics", key="plot_quality_btn"):
            try:
                fig = st.session_state.analyzer.plot_quality_metrics(
                    st.session_state.quality_metrics, 
                    metric_type=metric_to_plot
                )
                st.session_state.quality_plot = fig
                st.sidebar.success("Quality plot created!")
                
            except Exception as e:
                st.sidebar.error(f"Quality plot error: {e}")

# --- SINGLE-TRIAL ANALYSIS ---
st.sidebar.subheader("📈 Single-Trial Analysis")

if st.session_state.epochs_created:
    single_trial_type = st.sidebar.selectbox(
        "Feature type:",
        options=['erp', 'power', 'phase', 'pac'],
        key="single_trial_type"
    )
    
    if single_trial_type in ['power', 'phase', 'pac']:
        freq_min = st.sidebar.number_input("Frequency min (Hz)", value=8.0, key="st_freq_min")
        freq_max = st.sidebar.number_input("Frequency max (Hz)", value=13.0, key="st_freq_max")
        frequency_band = (freq_min, freq_max)
    else:
        frequency_band = None
    
    if st.sidebar.button("Extract Single-Trial Features", key="extract_single_trial_btn"):
        with st.spinner("Extracting single-trial features..."):
            try:
                single_trial_features = st.session_state.analyzer.extract_single_trial_features(
                    feature_type=single_trial_type,
                    frequency_band=frequency_band
                )
                st.session_state.single_trial_features = single_trial_features
                st.sidebar.success("Single-trial features extracted!")
                
            except Exception as e:
                st.sidebar.error(f"Single-trial extraction error: {e}")
    
    # Plot single-trial analysis
    if hasattr(st.session_state, 'single_trial_features'):
        plot_type = st.sidebar.selectbox(
            "Plot type:",
            options=['heatmap', 'scatter', 'distribution'],
            key="single_trial_plot_type"
        )
        
        if st.sidebar.button("Plot Single-Trial Analysis", key="plot_single_trial_btn"):
            try:
                fig = st.session_state.analyzer.plot_single_trial_analysis(
                    st.session_state.single_trial_features,
                    feature_type=single_trial_type,
                    plot_type=plot_type
                )
                st.session_state.single_trial_plot = fig
                st.sidebar.success("Single-trial plot created!")
                
            except Exception as e:
                st.sidebar.error(f"Single-trial plot error: {e}")

# --- COGNITIVE MODELING ---
st.sidebar.subheader("🧠 Cognitive Modeling")

if (hasattr(st.session_state, 'behavioral_loaded') and st.session_state.behavioral_loaded and 
    hasattr(st.session_state, 'single_trial_features')):
    
    model_type = st.sidebar.selectbox(
        "Model type:",
        options=['linear', 'logistic'],
        key="cognitive_model_type"
    )
    
    if hasattr(st.session_state.analyzer, 'behavioral_data'):
        behavioral_column = st.sidebar.selectbox(
            "Behavioral variable:",
            options=st.session_state.analyzer.behavioral_data.columns.tolist(),
            key="cognitive_behavioral_var"
        )
        
        if st.sidebar.button("Fit GLM Model", key="fit_glm_btn"):
            with st.spinner("Fitting GLM model..."):
                try:
                    behavioral_data = st.session_state.analyzer.behavioral_data[behavioral_column].values
                    glm_results = st.session_state.analyzer.fit_glm_model(
                        behavioral_data=behavioral_data,
                        neural_features=st.session_state.single_trial_features,
                        model_type=model_type
                    )
                    st.session_state.glm_results = glm_results
                    st.sidebar.success("GLM model fitted!")
                    
                except Exception as e:
                    st.sidebar.error(f"GLM fitting error: {e}")

# --- PRE-TRAINED MODELS ---
st.sidebar.subheader("🤖 Pre-trained Models")

pretrained_model_file = st.sidebar.file_uploader(
    "Upload pre-trained model:",
    type=["pkl", "joblib"],
    key="pretrained_model_uploader"
)

if pretrained_model_file is not None:
    try:
        # Save uploaded model temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(pretrained_model_file.name)[1]) as tmp_file:
            tmp_file.write(pretrained_model_file.getvalue())
            tmp_path = tmp_file.name
        
        model_type = st.sidebar.selectbox(
            "Model type:",
            options=['classifier', 'regressor', 'autoencoder'],
            key="pretrained_model_type"
        )
        
        if st.sidebar.button("Load Pre-trained Model", key="load_pretrained_btn"):
            with st.spinner("Loading pre-trained model..."):
                try:
                    model = st.session_state.analyzer.load_pretrained_model(
                        tmp_path, model_type=model_type
                    )
                    if model is not None:
                        st.sidebar.success("Pre-trained model loaded!")
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.sidebar.error(f"Model loading error: {e}")
    
    except Exception as e:
        st.sidebar.error(f"File upload error: {e}")

# Apply pre-trained model
if hasattr(st.session_state.analyzer, 'pretrained_model') and st.session_state.epochs_created:
    if st.sidebar.button("Apply Pre-trained Model", key="apply_pretrained_btn"):
        with st.spinner("Applying pre-trained model..."):
            try:
                results = st.session_state.analyzer.apply_pretrained_model(
                    data=None, feature_extraction=True
                )
                st.session_state.pretrained_results = results
                st.sidebar.success("Pre-trained model applied!")
                
            except Exception as e:
                st.sidebar.error(f"Model application error: {e}")

# --- MOVIE GENERATION ---
st.sidebar.subheader("📽️ Movie Generation")

if st.session_state.epochs_created:
    movie_data_type = st.sidebar.selectbox(
        "Data type:",
        options=['erp', 'power', 'topomap', 'pac'],
        key="movie_data_type"
    )
    
    fps = st.sidebar.number_input("FPS:", value=10, min_value=1, max_value=30, key="movie_fps")
    duration = st.sidebar.number_input("Duration (s):", value=5, min_value=1, max_value=30, key="movie_duration")
    
    # Add info about requirements
    if movie_data_type == 'power' and not st.session_state.tfr_computed:
        st.sidebar.warning("⚠️ Time-frequency analysis required for power movies")
    elif movie_data_type == 'pac' and not hasattr(st.session_state.analyzer, 'pac_results'):
        st.sidebar.warning("⚠️ PAC analysis required for PAC movies")
    
    if st.sidebar.button("Generate Movie", key="generate_movie_btn"):
        with st.spinner("Generating movie..."):
            try:
                output_path = f"neural_analysis_movie_{movie_data_type}.mp4"
                movie_path = st.session_state.analyzer.generate_time_series_movie(
                    data_type=movie_data_type,
                    output_path=output_path,
                    fps=fps,
                    duration=duration
                )
                if movie_path:
                    st.session_state.movie_path = movie_path
                    st.sidebar.success("Movie generated!")
                    
                    # Create download link
                    with open(movie_path, 'rb') as f:
                        movie_data = f.read()
                    
                    # Determine MIME type based on file extension
                    file_ext = os.path.splitext(movie_path)[1].lower()
                    mime_type = "video/mp4" if file_ext == ".mp4" else "image/gif"
                    
                    st.sidebar.download_button(
                        label="Download Movie",
                        data=movie_data,
                        file_name=os.path.basename(movie_path),
                        mime=mime_type
                    )
                else:
                    st.sidebar.error("Movie generation failed. Check console for details.")
                
            except Exception as e:
                st.sidebar.error(f"Movie generation error: {e}")

# --- AI-POWERED ERROR TRACKING ---
st.sidebar.subheader("🔍 AI Quality Analysis")

if st.sidebar.button("Analyze Data Quality", key="analyze_quality_btn"):
    with st.spinner("Analyzing data quality..."):
        try:
            quality_analysis = st.session_state.analyzer.analyze_data_quality()
            st.session_state.quality_analysis = quality_analysis
            st.sidebar.success("Quality analysis completed!")
            
        except Exception as e:
            st.sidebar.error(f"Quality analysis error: {e}")

# --- DATA PIPELINE FEATURES ---
if advanced_feature == "Data Pipeline" and PIPELINE_AVAILABLE:
    st.sidebar.header("📊 Data Pipeline")
    
    # Pipeline configuration
    st.sidebar.subheader("Pipeline Configuration")
    
    input_dir = st.sidebar.text_input("Input Directory", value="./data")
    output_dir = st.sidebar.text_input("Output Directory", value="./processed_data")
    
    file_patterns = st.sidebar.multiselect(
        "File Patterns",
        options=['*.edf', '*.csv', '*.mat', '*.fif'],
        default=['*.edf', '*.csv']
    )
    
    preprocessing_steps = st.sidebar.multiselect(
        "Preprocessing Steps",
        options=['filter', 'notch', 'clean'],
        default=['filter', 'notch']
    )
    
    analysis_types = st.sidebar.multiselect(
        "Analysis Types",
        options=['spectral', 'erp', 'statistics'],
        default=['spectral', 'erp']
    )
    
    if st.sidebar.button("Create Pipeline", key="create_pipeline_btn"):
        try:
            config = PipelineConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                file_patterns=file_patterns,
                preprocessing_steps=preprocessing_steps,
                analysis_types=analysis_types,
                parallel_processing=True,
                max_workers=4
            )
            
            pipeline = DataPipeline(config)
            st.session_state.data_pipeline = pipeline
            st.sidebar.success("Data pipeline created!")
            
        except Exception as e:
            st.sidebar.error(f"Pipeline creation error: {e}")
    
    # Run pipeline
    if hasattr(st.session_state, 'data_pipeline'):
        if st.sidebar.button("Run Pipeline", key="run_pipeline_btn"):
            with st.spinner("Running data pipeline..."):
                try:
                    results = st.session_state.data_pipeline.run_pipeline()
                    st.session_state.pipeline_results = results
                    st.sidebar.success(f"Pipeline completed! Processed {len(results)} files.")
                    
                except Exception as e:
                    st.sidebar.error(f"Pipeline execution error: {e}")
    
    # Data validation
    st.sidebar.subheader("Data Validation")
    validation_files = st.sidebar.file_uploader(
        "Upload files for validation",
        type=["edf", "csv", "mat", "fif"],
        accept_multiple_files=True,
        key="validation_files"
    )
    
    if validation_files and st.sidebar.button("Validate Files", key="validate_files_btn"):
        with st.spinner("Validating files..."):
            try:
                file_paths = []
                for uploaded_file in validation_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        file_paths.append(tmp_file.name)
                
                validation_results = DataValidator.validate_dataset_consistency(file_paths)
                st.session_state.validation_results = validation_results
                
                # Clean up temp files
                for file_path in file_paths:
                    os.unlink(file_path)
                
                st.sidebar.success("Validation completed!")
                
            except Exception as e:
                st.sidebar.error(f"Validation error: {e}")

# --- WORKFLOW AUTOMATION ---
if advanced_feature == "Workflow Automation" and WORKFLOW_AVAILABLE:
    st.sidebar.header("🔄 Workflow Automation")
    
    # Workflow manager
    workflow_manager = WorkflowManager()
    available_workflows = workflow_manager.list_workflows()
    
    st.sidebar.subheader("Available Workflows")
    if available_workflows:
        selected_workflow = st.sidebar.selectbox(
            "Select Workflow",
            options=available_workflows,
            key="workflow_select"
        )
        
        if st.sidebar.button("Load Workflow", key="load_workflow_btn"):
            try:
                config = workflow_manager.load_workflow_config(selected_workflow)
                st.session_state.current_workflow = config
                st.sidebar.success(f"Workflow '{selected_workflow}' loaded!")
                
            except Exception as e:
                st.sidebar.error(f"Workflow loading error: {e}")
    
    # Create new workflow
    st.sidebar.subheader("Create New Workflow")
    workflow_name = st.sidebar.text_input("Workflow Name", key="new_workflow_name")
    workflow_description = st.sidebar.text_area("Description", key="new_workflow_desc")
    
    if st.sidebar.button("Create Standard Workflow", key="create_workflow_btn"):
        try:
            config = workflow_manager.create_standard_workflow(
                name=workflow_name,
                input_paths=["./data"],
                output_path="./workflow_output"
            )
            workflow_manager.save_workflow_config(config)
            st.sidebar.success(f"Workflow '{workflow_name}' created!")
            
        except Exception as e:
            st.sidebar.error(f"Workflow creation error: {e}")
    
    # Execute workflow
    if hasattr(st.session_state, 'current_workflow'):
        if st.sidebar.button("Execute Workflow", key="execute_workflow_btn"):
            with st.spinner("Executing workflow..."):
                try:
                    executor = WorkflowExecutor(st.session_state.current_workflow)
                    workflow_result = executor.execute_workflow()
                    st.session_state.workflow_result = workflow_result
                    st.sidebar.success("Workflow executed successfully!")
                    
                except Exception as e:
                    st.sidebar.error(f"Workflow execution error: {e}")

# --- DATA MANAGEMENT ---
if advanced_feature == "Data Management" and DATA_MANAGEMENT_AVAILABLE:
    st.sidebar.header("🗂️ Data Management")
    
    # BIDS management
    st.sidebar.subheader("BIDS Format")
    bids_root = st.sidebar.text_input("BIDS Root Directory", value="./bids_dataset")
    
    if st.sidebar.button("Initialize BIDS", key="init_bids_btn"):
        try:
            bids_manager = BIDSManager(bids_root)
            st.session_state.bids_manager = bids_manager
            st.sidebar.success("BIDS dataset initialized!")
            
        except Exception as e:
            st.sidebar.error(f"BIDS initialization error: {e}")
    
    # Add participant
    if hasattr(st.session_state, 'bids_manager'):
        st.sidebar.subheader("Add Participant")
        participant_id = st.sidebar.text_input("Participant ID", key="participant_id")
        age = st.sidebar.number_input("Age", min_value=0, max_value=120, key="participant_age")
        sex = st.sidebar.selectbox("Sex", options=['M', 'F', 'O'], key="participant_sex")
        
        if st.sidebar.button("Add Participant", key="add_participant_btn"):
            try:
                participant = BIDSParticipant(
                    participant_id=participant_id,
                    age=age,
                    sex=sex
                )
                st.session_state.bids_manager.add_participant(participant)
                st.sidebar.success(f"Participant {participant_id} added!")
                
            except Exception as e:
                st.sidebar.error(f"Participant addition error: {e}")
    
    # Database management
    st.sidebar.subheader("Database")
    if st.sidebar.button("Initialize Database", key="init_db_btn"):
        try:
            db_manager = DatabaseManager()
            st.session_state.db_manager = db_manager
            st.sidebar.success("Database initialized!")
            
        except Exception as e:
            st.sidebar.error(f"Database initialization error: {e}")
    
    # Data organization
    st.sidebar.subheader("Data Organization")
    organize_method = st.sidebar.selectbox(
        "Organization Method",
        options=['by_participant', 'by_date', 'by_type'],
        key="organize_method"
    )
    
    organize_files = st.sidebar.file_uploader(
        "Upload files to organize",
        type=["edf", "csv", "mat", "fif"],
        accept_multiple_files=True,
        key="organize_files"
    )
    
    if organize_files and st.sidebar.button("Organize Files", key="organize_files_btn"):
        with st.spinner("Organizing files..."):
            try:
                file_paths = []
                for uploaded_file in organize_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        file_paths.append(tmp_file.name)
                
                organizer = DataOrganizer()
                
                if organize_method == 'by_participant':
                    # Simple mapping for demo
                    participant_mapping = {os.path.basename(f): f"p{i+1}" for i, f in enumerate(file_paths)}
                    organizer.organize_by_participant(file_paths, participant_mapping)
                elif organize_method == 'by_date':
                    organizer.organize_by_date(file_paths)
                elif organize_method == 'by_type':
                    organizer.organize_by_type(file_paths)
                
                # Clean up temp files
                for file_path in file_paths:
                    os.unlink(file_path)
                
                st.sidebar.success("Files organized successfully!")
                
            except Exception as e:
                st.sidebar.error(f"Organization error: {e}")

# --- ENHANCED UX ---
if advanced_feature == "Enhanced UX" and UX_AVAILABLE:
    st.sidebar.header("🎨 Enhanced UX")
    
    # Initialize enhanced UX
    if 'enhanced_ux_initialized' not in st.session_state:
        shortcut_manager = initialize_enhanced_ux()
        st.session_state.shortcut_manager = shortcut_manager
        st.session_state.enhanced_ux_initialized = True
    
    # Render enhanced sidebar
    render_enhanced_sidebar()
    
    # Keyboard shortcuts help
    if st.sidebar.button("Keyboard Shortcuts Help", key="shortcuts_help_btn"):
        st.session_state.show_shortcuts_help = True
    
    # Dashboard customization
    if st.sidebar.button("Customize Dashboard", key="customize_dashboard_btn"):
        st.session_state.show_dashboard_customization = True



# --- AI ASSISTANT ---
st.sidebar.header("🤖 AI Assistant")

# AI chat interface
user_question = st.sidebar.text_area(
    "Ask me about neural data analysis:",
    placeholder="e.g., How do I preprocess my EEG data?",
    key="ai_question_input",
    height=100
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Ask AI", key="ask_ai_btn"):
        if user_question.strip():
            with st.spinner("🤖 Thinking..."):
                response = ai_assistant.get_response(user_question)
                st.session_state.ai_response = response
                st.session_state.ai_question_text = user_question
        else:
            st.sidebar.warning("Please enter a question.")

with col2:
    if st.button("Get Help", key="get_help_btn"):
        st.session_state.ai_response = ai_assistant.get_help()
        st.session_state.ai_question_text = "Help"

# Display AI response
if hasattr(st.session_state, 'ai_response'):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 AI Response:**")
    st.sidebar.info(st.session_state.ai_response)

# Quick tips
if st.sidebar.button("Get Random Tip", key="get_tip_btn"):
    tip = ai_assistant.get_tips()
    st.sidebar.success(f"💡 **Tip:** {tip}")

# --- REAL-TIME STREAMING ---
st.sidebar.subheader("⏱️ Real-Time Streaming")

stream_type = st.sidebar.selectbox(
    "Stream type:",
    options=['simulation'],
    key="stream_type"
)

if st.sidebar.button("Setup Real-Time Streaming", key="setup_streaming_btn"):
    with st.spinner("Setting up real-time streaming..."):
        try:
            success = st.session_state.analyzer.setup_realtime_streaming(stream_type=stream_type)
            if success:
                st.session_state.realtime_enabled = True
                st.sidebar.success("Real-time streaming enabled!")
            else:
                st.sidebar.warning("Real-time streaming setup failed")
                
        except Exception as e:
            st.sidebar.error(f"Streaming setup error: {e}")

# --- MAIN CONTENT AREA FOR ADVANCED FEATURES ---

# Display signal quality metrics
if hasattr(st.session_state, 'quality_metrics'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Signal Quality Metrics</h2>", unsafe_allow_html=True)
    
    quality_metrics = st.session_state.quality_metrics
    
    # Create summary table
    summary_data = []
    for ch_name, metrics in quality_metrics.items():
        row = {'Channel': ch_name}
        for metric_name, value in metrics.items():
            if metric_name.startswith('mean_'):
                row[metric_name.replace('mean_', '')] = f"{value:.3f}"
        summary_data.append(row)
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        st.dataframe(df)
        
        # Download quality metrics
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Quality Metrics (CSV)",
            data=csv,
            file_name="signal_quality_metrics.csv",
            mime="text/csv"
        )
    
    # Display quality plot
    if hasattr(st.session_state, 'quality_plot'):
        st.pyplot(st.session_state.quality_plot)

# Display single-trial analysis
if hasattr(st.session_state, 'single_trial_features'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Single-Trial Analysis</h2>", unsafe_allow_html=True)
    
    single_trial_features = st.session_state.single_trial_features
    
    # Create summary statistics
    summary_stats = []
    for ch_name, features in single_trial_features.items():
        for feature_name, values in features.items():
            summary_stats.append({
                'Channel': ch_name,
                'Feature': feature_name,
                'Mean': f"{np.mean(values):.3f}",
                'Std': f"{np.std(values):.3f}",
                'Min': f"{np.min(values):.3f}",
                'Max': f"{np.max(values):.3f}"
            })
    
    if summary_stats:
        df = pd.DataFrame(summary_stats)
        st.dataframe(df)
        
        # Download single-trial features
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Single-Trial Features (CSV)",
            data=csv,
            file_name="single_trial_features.csv",
            mime="text/csv"
        )
    
    # Display single-trial plot
    if hasattr(st.session_state, 'single_trial_plot'):
        st.pyplot(st.session_state.single_trial_plot)

# Display GLM results
if hasattr(st.session_state, 'glm_results'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Cognitive Modeling Results</h2>", unsafe_allow_html=True)
    
    glm_results = st.session_state.glm_results
    
    # Display model metrics
    metrics = glm_results['metrics']
    col1, col2 = st.columns(2)
    with col1:
        for metric_name, value in metrics.items():
            st.metric(metric_name.upper(), f"{value:.3f}")
    
    # Display feature importance
    if glm_results['feature_importance'] is not None:
        with col2:
            st.subheader("Feature Importance")
            feature_importance_df = pd.DataFrame({
                'Feature': glm_results['feature_names'],
                'Importance': glm_results['feature_importance']
            }).sort_values('Importance', ascending=False)
            
            st.dataframe(feature_importance_df.head(10))

# Display pre-trained model results
if hasattr(st.session_state, 'pretrained_results'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Pre-trained Model Results</h2>", unsafe_allow_html=True)
    
    pretrained_results = st.session_state.pretrained_results
    
    st.subheader(f"Model Type: {pretrained_results['model_type']}")
    
    # Display predictions
    if 'predictions' in pretrained_results:
        predictions = pretrained_results['predictions']
        st.write(f"Number of predictions: {len(predictions)}")
        
        # Create prediction histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(predictions, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title('Prediction Distribution')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

# Display quality analysis
if hasattr(st.session_state, 'quality_analysis'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>Data Quality Analysis</h2>", unsafe_allow_html=True)
    
    quality_analysis = st.session_state.quality_analysis
    
    # Display quality score (if available)
    if 'quality_score' in quality_analysis:
        quality_score = quality_analysis['quality_score']
        st.metric("Overall Quality Score", f"{quality_score}/100")
    
    # Display warnings
    if quality_analysis.get('warnings'):
        st.subheader("⚠️ Warnings")
        for warning in quality_analysis['warnings']:
            st.warning(warning)
    
    # Display recommendations
    if quality_analysis.get('recommendations'):
        st.subheader("💡 Recommendations")
        for rec in quality_analysis['recommendations']:
            st.success(rec)



# --- END WATERMARK ---
st.markdown("---")
# --- AI CONVERSATION HISTORY ---
if hasattr(st.session_state, 'ai_response'):
    st.markdown("---")
    st.markdown("<h2 class='section-header'>🤖 AI Assistant Conversation</h2>", unsafe_allow_html=True)
    
    st.markdown(f"**Your Question:** {st.session_state.ai_question_text}")
    st.markdown(f"**AI Response:** {st.session_state.ai_response}")
    
    # Clear conversation button
    if st.button("Clear Conversation", key="clear_conversation_btn"):
        if hasattr(st.session_state, 'ai_response'):
            del st.session_state.ai_response
        if hasattr(st.session_state, 'ai_question_text'):
            del st.session_state.ai_question_text
        st.rerun()

st.markdown('<div class="end-watermark">🧠 Neural Data Analysis Pipeline v2.0 | Built with Streamlit & Simple AI | Paarth Goela (2023A7PS0006H)</div>', unsafe_allow_html=True)

st.markdown("""
<style>
/* Progress bar styles */
.progress-step {
    background: #23272f;
    border-radius: 8px;
    margin-bottom: 10px;
    padding: 12px 20px;
    color: #fff;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
}
.progress-step.completed {
    background: #2e7d32 !important; /* Vibrant green */
    color: #fff !important;
}
.progress-step.incomplete {
    background: #23272f !important;
    color: #90a4ae !important;
}
.progress-step .icon {
    margin-right: 12px;
    font-size: 1.3em;
}
.section-title {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    color: #fff !important;
}
.upload-box {
    border: 2px dashed #90caf9 !important;
    border-radius: 12px !important;
    background: #23272f !important;
    color: #fff !important;
    padding: 24px !important;
    margin-bottom: 1rem !important;
    transition: border-color 0.2s;
}
.upload-box:hover {
    border-color: #42a5f5 !important;
}
</style>
""", unsafe_allow_html=True)

# Example usage for section title
st.markdown('<div class="section-title">1. Load Data</div>', unsafe_allow_html=True)
# Example usage for upload box
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload EEG/LFP file (.edf, .fif, .set)", type=["edf", "fif", "set"])
st.markdown('</div>', unsafe_allow_html=True)
# Example usage for progress steps
st.markdown('<div class="section-title">Analysis Progress</div>', unsafe_allow_html=True)
st.markdown('<div class="progress-step completed"><span class="icon">✅</span>Data Loaded</div>', unsafe_allow_html=True)
st.markdown('<div class="progress-step incomplete"><span class="icon">⏳</span>Preprocessed</div>', unsafe_allow_html=True)
st.markdown('<div class="progress-step incomplete"><span class="icon">⏳</span>Epochs Created</div>', unsafe_allow_html=True)
st.markdown('<div class="progress-step incomplete"><span class="icon">⏳</span>TFR Computed</div>', unsafe_allow_html=True)