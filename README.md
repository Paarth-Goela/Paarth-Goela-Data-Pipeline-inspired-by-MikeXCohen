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

### 4. Advanced Features

#### Signal Quality Metrics
```
# Compute comprehensive quality metrics
quality_metrics = analyzer.compute_signal_quality_metrics(
    metrics=['variance', 'entropy', 'kurtosis', 'snr', 'line_noise']
)

# Plot quality metrics
fig = analyzer.plot_quality_metrics(quality_metrics, metric_type='variance')
```

#### Single-Trial Analysis
```
# Extract single-trial features
single_trial_features = analyzer.extract_single_trial_features(
    feature_type='erp',
    time_window=(0.1, 0.3)
)

# Visualize single-trial data
fig = analyzer.plot_single_trial_analysis(
    single_trial_features,
    plot_type='heatmap'
)
```

#### Cognitive Modeling
```
# Fit GLM model
glm_results = analyzer.fit_glm_model(
    behavioral_data=reaction_times,
    neural_features=single_trial_features,
    model_type='linear'
)

# Access model results
r2_score = glm_results['metrics']['r2']
feature_importance = glm_results['feature_importance']
```

#### Pre-trained Models
```
# Load pre-trained model
model = analyzer.load_pretrained_model('model.joblib', 'classifier')

# Apply to new data
results = analyzer.apply_pretrained_model(data, feature_extraction=True)
```

#### Movie Generation
```
# Generate animated time-series movie
movie_path = analyzer.generate_time_series_movie(
    data_type='erp',
    output_path='analysis_movie.mp4',
    fps=10,
    duration=5
)
```

#### Statistical Analysis
```
# Run comprehensive statistical tests
stats_results = analyzer.statistical_analysis(
    test_type='parametric',
    alpha=0.05,
    correction='fdr_bh'
)

# Cluster permutation test
cluster_results = analyzer.cluster_permutation_test(
    n_permutations=1000,
    alpha=0.05
)
```

#### Custom Frequency Bands
```
# Define custom frequency bands
custom_bands = {
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 80),
    'theta': (4, 8)
}
analyzer.define_custom_frequency_bands(custom_bands)

# Analyze custom bands
results = analyzer.analyze_custom_bands(method='power')
```

#### Behavioral Data Integration
```
# Load behavioral data
analyzer.load_behavioral_data('behavioral_data.csv')

# Correlate neural measures with behavioral variables
correlations = analyzer.correlate_neural_behavioral(
    neural_measure='power',
    behavioral_column='reaction_time'
)
```

#### Export & Reproducibility
```
# Export complete analysis pipeline
export_dir = analyzer.export_analysis_pipeline()

# Export to BIDS format
bids_dir = analyzer.export_to_bids()
```

### 5. Interactive Features
- **Interactive Plots**: Zoom, pan, and explore time-frequency data
- **Dashboard**: Real-time metrics and analysis summaries
- **Assistant**: Local-only, no external API

## 🔧 Configuration

### Custom Settings
- **Frequency Bands**: Modify default frequency ranges
- **Statistical Parameters**: Adjust significance levels and correction methods
- **Visualization**: Customize plot styles and colors

## 📁 File Structure

```
PythonProject/
├── streamlit_app.py          # Main Streamlit application
├── neural_backend.py         # Core analysis backend
├── requirements.txt          # Python dependencies
├── simple_ai_assistant.py    # Local AI assistant (no external API)
├── deploy.py                 # Deployment helper
├── deployment_guide.md       # Deployment instructions
├── README.md                 # This file
```

## 🧪 Testing

### Test PAC Functionality
```bash
python test_pac.py
```

### Test Advanced Features
```bash
python test_advanced_features.py
```

## 📈 Example Workflows

### Basic EEG Analysis
1. Load sample data or upload EEG file
2. Preprocess data (filter, remove artifacts)
3. Create epochs around events
4. Perform time-frequency analysis
5. Compute PAC measures
6. Run statistical tests
7. Generate report

### Advanced Analysis with Behavioral Data
1. Load neural and behavioral data
2. Preprocess both datasets
3. Define custom frequency bands
4. Extract single-trial features
5. Correlate neural measures with behavioral variables
6. Fit cognitive models (GLM/DDM)
7. Perform statistical analysis
8. Create interactive visualizations
9. Export results in BIDS format

### Quality-Focused Analysis
1. Load and preprocess data
2. Compute signal quality metrics
3. Analyze data quality
4. Apply quality-based filtering
5. Extract single-trial features
6. Generate quality report
7. Create animated visualizations

### Clinical Screening Pipeline
1. Load pre-trained clinical model
2. Process new patient data
3. Extract relevant features
4. Apply pre-trained classifier
5. Generate clinical report
6. Export results for medical records

### Batch Processing
1. Upload multiple files
2. Set preprocessing and analysis parameters
3. Run batch processing with quality monitoring
4. Review batch results
5. Generate comprehensive report
6. Export pipeline for reproducibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Mike X Cohen**: For the foundational analysis approach
- **MNE-Python**: For the excellent EEG/MEG analysis framework
- **Streamlit**: For the interactive web interface

## 📞 Support

For questions, issues, or feature requests:
1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information

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