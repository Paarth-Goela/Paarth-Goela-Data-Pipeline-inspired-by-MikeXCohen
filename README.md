# 🧠 Neural Data Analysis Pipeline

A comprehensive neural data analysis pipeline based on Mike X Cohen's approach, featuring advanced EEG/LFP analysis, machine learning, and interactive visualization capabilities.

## 🌟 Features

### Core Analysis
- **Data Loading & Preprocessing**: Support for EDF, FIF, and EEGLAB formats
- **Artifact Detection & Removal**: ICA-based artifact removal and bad channel/epoch marking
- **Time-Frequency Analysis**: Morlet wavelet-based time-frequency decomposition
- **Phase-Amplitude Coupling (PAC)**: Multiple PAC methods (Tort, Canolty, Ozkurt, PLV)
- **Event-Related Potentials (ERPs)**: ERP computation, visualization, and statistical comparison
- **Connectivity Analysis**: Coherence, Phase Locking Value (PLV), and Granger causality
- **Source Localization**: Forward/inverse modeling with visualization

### Advanced Statistical Analysis
- **Parametric & Nonparametric Tests**: T-tests, ANOVA, Mann-Whitney U, Kruskal-Wallis
- **Multiple Comparison Correction**: FDR, Bonferroni, Holm methods
- **Cluster Permutation Tests**: Time-frequency cluster-based statistics
- **Channel-wise Analysis**: Comprehensive statistical testing across all channels

### Signal Quality Assessment
- **Auto-Quality Metrics**: Variance, entropy, kurtosis, SNR, line noise detection
- **Trial-wise Analysis**: Single-trial feature extraction and visualization
- **Quality Scoring**: Data quality assessment and recommendations
- **Real-time Monitoring**: Live quality metrics for streaming data

### Single-Trial Analysis
- **Trial-wise Features**: ERP amplitude, power, phase, PAC for each trial
- **Visualization**: Heatmaps, scatter plots, and distribution analysis
- **Trend Analysis**: Trial-by-trial evolution and learning effects
- **Export Capabilities**: Comprehensive single-trial data export

### Cognitive Modeling Integration
- **GLM Analysis**: Generalized Linear Models for neural-behavioral correlation
- **Drift Diffusion Modeling (DDM)**: Decision-making model fitting
- **Pre-trained Model Integration**: Apply ML models to new data

### Export & Reproducibility
- **BIDS Export**: Export results in BIDS format
- **Reproducibility Scripts**: Auto-generated scripts for analysis reproduction
- **Comprehensive Documentation**: Detailed analysis logs and metadata

### Interactive Exploration
- **Interactive Plots**: Plotly-based interactive visualizations
- **Dashboard**: Real-time analysis metrics and summaries
- **Custom Visualizations**: User-defined plotting options

### Time-Series Movie Generation
- **Animated Visualizations**: ERP, power, and topomap animations
- **Customizable Parameters**: FPS, duration, and data type selection
- **Export Options**: MP4 format for presentations and publications
- **Real-time Preview**: Live preview of generated animations

### Real-Time Streaming Support
- **BCI-Ready**: Lab Streaming Layer (LSL) integration
- **Live Processing**: Real-time data analysis and visualization
- **Simulation Mode**: Test real-time features with simulated data
- **Neurofeedback**: Ready for neurofeedback applications

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PythonProject
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and navigate to the provided URL (usually `http://localhost:8501`)

## 📊 Usage Guide

### 1. Data Loading
- **Sample Data**: Use built-in MNE sample data for testing
- **Upload Files**: Support for EDF, FIF, and EEGLAB formats
- **Batch Upload**: Process multiple files simultaneously

### 2. Preprocessing
- **Filtering**: Bandpass, notch, and high-pass filtering
- **Artifact Removal**: ICA-based artifact detection and removal
- **Bad Channel/Epoch Marking**: Manual or automatic bad data identification

### 3. Analysis Pipeline
- **Epoching**: Create epochs around events or artificial triggers
- **Time-Frequency Analysis**: Compute power spectra and time-frequency representations
- **PAC Analysis**: Multiple phase-amplitude coupling methods
- **ERP Analysis**: Event-related potential computation and visualization



## 🔄 Version History

### v2.1.0 (Current)
- ✅ Signal quality metrics (variance, entropy, kurtosis, SNR)
- ✅ Single-trial analysis with visualization
- ✅ Cognitive modeling (GLM, DDM)
- ✅ Pre-trained model integration
- ✅ Time-series movie generation
- ✅ Real-time streaming support
- ✅ Enhanced quality assessment

### v2.0.0
- ✅ Statistical analysis with multiple comparison corrections
- ✅ Custom frequency band analysis
- ✅ Behavioral data integration
- ✅ Enhanced export and reproducibility features
- ✅ Interactive exploration capabilities
- ✅ Cluster permutation tests
- ✅ BIDS format export
- ✅ Comprehensive dashboard

### v1.0.0
- ✅ Basic EEG/LFP analysis pipeline
- ✅ PAC analysis with multiple methods
- ✅ Machine learning classification
- ✅ Streamlit web interface

---

**Happy analyzing! 🧠✨** 
