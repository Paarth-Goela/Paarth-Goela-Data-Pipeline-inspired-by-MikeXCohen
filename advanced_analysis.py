"""
Advanced Analysis Methods
Source localization, connectivity analysis, and non-linear dynamics
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import mne
from scipy import signal, stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyzer:
    """
    Advanced analysis methods for EEG data
    """
    
    def __init__(self):
        self.source_results = None
        self.connectivity_results = None
        self.nonlinear_results = None
    
    def compute_source_localization(self, epochs, method='dspm', lambda2=1.0/9.0):
        """
        Compute source localization using MNE methods
        """
        try:
            # Create forward model
            info = epochs.info
            src = mne.setup_source_space('sample', spacing='oct6', add_dist=False)
            
            # Create forward solution
            fwd = mne.make_forward_solution(info, trans=None, src=src, bem=None)
            
            # Create noise covariance
            noise_cov = mne.compute_covariance(epochs, tmax=0)
            
            # Compute inverse solution
            if method == 'dspm':
                inv = mne.beamformer.make_dspm_solution(fwd, noise_cov, epochs.info, 
                                                      lambda2=lambda2, pick_ori='max-power')
            elif method == 'sloreta':
                inv = mne.beamformer.make_lcmv_solution(fwd, noise_cov, epochs.info, 
                                                      lambda2=lambda2, pick_ori='max-power')
            else:
                inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov)
            
            # Apply inverse to epochs
            stc = mne.beamformer.apply_dspm(epochs.average(), inv)
            
            self.source_results = {
                'stc': stc,
                'method': method,
                'vertices': stc.vertices,
                'times': stc.times,
                'data': stc.data
            }
            
            return True
            
        except Exception as e:
            st.error(f"Source localization failed: {e}")
            return False
    
    def compute_connectivity_analysis(self, epochs, method='coh', freqs=None):
        """
        Compute connectivity between channels
        """
        try:
            if freqs is None:
                freqs = np.logspace(np.log10(6), np.log10(35), 20)
            
            # Get data
            data = epochs.get_data()
            n_epochs, n_channels, n_times = data.shape
            
            # Initialize connectivity matrix
            connectivity_matrix = np.zeros((n_channels, n_channels, len(freqs)))
            
            if method == 'coh':
                # Coherence
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        for f_idx, freq in enumerate(freqs):
                            # Compute coherence for each epoch
                            cohs = []
                            for epoch in range(n_epochs):
                                freqs_coh, coh = signal.coherence(
                                    data[epoch, i, :], 
                                    data[epoch, j, :], 
                                    fs=epochs.info['sfreq'],
                                    nperseg=min(256, n_times//4)
                                )
                                # Find closest frequency
                                freq_idx = np.argmin(np.abs(freqs_coh - freq))
                                cohs.append(coh[freq_idx])
                            
                            connectivity_matrix[i, j, f_idx] = np.mean(cohs)
                            connectivity_matrix[j, i, f_idx] = connectivity_matrix[i, j, f_idx]
            
            elif method == 'plv':
                # Phase Locking Value
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        for f_idx, freq in enumerate(freqs):
                            # Filter data at specific frequency
                            b, a = signal.butter(4, [freq-1, freq+1], btype='band', fs=epochs.info['sfreq'])
                            
                            plvs = []
                            for epoch in range(n_epochs):
                                # Filter signals
                                sig1_filt = signal.filtfilt(b, a, data[epoch, i, :])
                                sig2_filt = signal.filtfilt(b, a, data[epoch, j, :])
                                
                                # Compute phases
                                phase1 = np.angle(signal.hilbert(sig1_filt))
                                phase2 = np.angle(signal.hilbert(sig2_filt))
                                
                                # Compute PLV
                                plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
                                plvs.append(plv)
                            
                            connectivity_matrix[i, j, f_idx] = np.mean(plvs)
                            connectivity_matrix[j, i, f_idx] = connectivity_matrix[i, j, f_idx]
            
            elif method == 'granger':
                # Granger Causality (simplified)
                for i in range(n_channels):
                    for j in range(i+1, n_channels):
                        for f_idx, freq in enumerate(freqs):
                            # Simplified Granger causality
                            gc_values = []
                            for epoch in range(n_epochs):
                                # Use correlation as proxy for Granger causality
                                corr = np.corrcoef(data[epoch, i, :], data[epoch, j, :])[0, 1]
                                gc_values.append(np.abs(corr))
                            
                            connectivity_matrix[i, j, f_idx] = np.mean(gc_values)
                            connectivity_matrix[j, i, f_idx] = connectivity_matrix[i, j, f_idx]
            
            self.connectivity_results = {
                'matrix': connectivity_matrix,
                'method': method,
                'frequencies': freqs,
                'channels': epochs.ch_names
            }
            
            return True
            
        except Exception as e:
            st.error(f"Connectivity analysis failed: {e}")
            return False
    
    def compute_nonlinear_dynamics(self, epochs, methods=['entropy', 'lyapunov', 'fractal']):
        """
        Compute non-linear dynamics measures
        """
        try:
            data = epochs.get_data()
            n_epochs, n_channels, n_times = data.shape
            
            results = {}
            
            if 'entropy' in methods:
                # Sample Entropy
                entropy_values = np.zeros((n_epochs, n_channels))
                for epoch in range(n_epochs):
                    for ch in range(n_channels):
                        entropy_values[epoch, ch] = self._sample_entropy(data[epoch, ch, :])
                
                results['entropy'] = {
                    'values': entropy_values,
                    'mean': np.mean(entropy_values, axis=0),
                    'std': np.std(entropy_values, axis=0)
                }
            
            if 'lyapunov' in methods:
                # Lyapunov Exponent (simplified)
                lyap_values = np.zeros((n_epochs, n_channels))
                for epoch in range(n_epochs):
                    for ch in range(n_channels):
                        lyap_values[epoch, ch] = self._lyapunov_exponent(data[epoch, ch, :])
                
                results['lyapunov'] = {
                    'values': lyap_values,
                    'mean': np.mean(lyap_values, axis=0),
                    'std': np.std(lyap_values, axis=0)
                }
            
            if 'fractal' in methods:
                # Fractal Dimension (Higuchi method)
                fractal_values = np.zeros((n_epochs, n_channels))
                for epoch in range(n_epochs):
                    for ch in range(n_channels):
                        fractal_values[epoch, ch] = self._fractal_dimension(data[epoch, ch, :])
                
                results['fractal'] = {
                    'values': fractal_values,
                    'mean': np.mean(fractal_values, axis=0),
                    'std': np.std(fractal_values, axis=0)
                }
            
            self.nonlinear_results = results
            return True
            
        except Exception as e:
            st.error(f"Non-linear analysis failed: {e}")
            return False
    
    def _sample_entropy(self, signal, m=2, r=0.2):
        """Compute sample entropy"""
        N = len(signal)
        r = r * np.std(signal)
        
        # Count matches
        A = 0  # m+1 point matches
        B = 0  # m point matches
        
        for i in range(N-m):
            for j in range(i+1, N-m):
                # Check m-point template
                if np.max(np.abs(signal[i:i+m] - signal[j:j+m])) <= r:
                    B += 1
                    # Check m+1-point template
                    if np.abs(signal[i+m] - signal[j+m]) <= r:
                        A += 1
        
        if B == 0:
            return np.nan
        
        return -np.log(A / B)
    
    def _lyapunov_exponent(self, signal, m=2, tau=1):
        """Compute Lyapunov exponent (simplified)"""
        N = len(signal)
        if N < m * tau + 1:
            return np.nan
        
        # Create delay vectors
        vectors = []
        for i in range(N - m * tau):
            vector = [signal[i + j * tau] for j in range(m)]
            vectors.append(vector)
        
        vectors = np.array(vectors)
        
        # Compute distances and find nearest neighbors
        distances = pdist(vectors)
        if len(distances) == 0:
            return np.nan
        
        # Simplified Lyapunov estimate
        return np.mean(np.log(distances + 1e-10))
    
    def _fractal_dimension(self, signal, k_max=8):
        """Compute fractal dimension using Higuchi method"""
        N = len(signal)
        k_values = range(1, min(k_max, N//4))
        
        L_k = []
        for k in k_values:
            L_m = []
            for m in range(k):
                # Create subseries
                indices = range(m, N, k)
                if len(indices) < 2:
                    continue
                
                # Compute length
                length = 0
                for i in range(len(indices)-1):
                    length += abs(signal[indices[i+1]] - signal[indices[i]])
                
                if length > 0:
                    L_m.append(length * (N-1) / (k**2 * len(indices)))
            
            if L_m:
                L_k.append(np.mean(L_m))
        
        if len(L_k) < 2:
            return np.nan
        
        # Fit line to log-log plot
        k_log = np.log(1/np.array(k_values))
        L_log = np.log(L_k)
        
        slope, _, _, _, _ = stats.linregress(k_log, L_log)
        return slope

def create_advanced_analysis_interface():
    """Create Streamlit interface for advanced analysis"""
    st.markdown("## 🔬 Advanced Analysis Methods")
    
    # Initialize analyzer
    if 'advanced_analyzer' not in st.session_state:
        st.session_state.advanced_analyzer = AdvancedAnalyzer()
    
    analyzer = st.session_state.advanced_analyzer
    
    # Check if we have data
    if not hasattr(st.session_state, 'analyzer') or st.session_state.analyzer.epochs is None:
        st.warning("Please load and preprocess data first in the main analysis section.")
        return
    
    epochs = st.session_state.analyzer.epochs
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Source Localization", "Connectivity Analysis", "Non-linear Dynamics"],
        key="advanced_analysis_type"
    )
    
    if analysis_type == "Source Localization":
        st.markdown("### 🧠 Source Localization")
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "Method:",
                ["dspm", "sloreta", "mne"],
                key="source_method"
            )
        
        with col2:
            lambda2 = st.slider(
                "Regularization (λ²):",
                min_value=0.01,
                max_value=1.0,
                value=0.11,
                step=0.01,
                key="source_lambda"
            )
        
        if st.button("Compute Source Localization"):
            with st.spinner("Computing source localization..."):
                success = analyzer.compute_source_localization(epochs, method, lambda2)
                
                if success:
                    st.success("Source localization completed!")
                    
                    # Display results
                    if analyzer.source_results:
                        results = analyzer.source_results
                        
                        # Create 3D source plot
                        fig = go.Figure()
                        
                        # Plot source activity at peak time
                        peak_time_idx = np.argmax(np.max(np.abs(results['data']), axis=0))
                        peak_data = results['data'][:, peak_time_idx]
                        
                        # Create 3D scatter plot
                        fig.add_trace(go.Scatter3d(
                            x=results['vertices'][0],
                            y=results['vertices'][1],
                            z=results['vertices'][2],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=peak_data,
                                colorscale='Viridis',
                                colorbar=dict(title="Source Activity")
                            ),
                            name="Source Activity"
                        ))
                        
                        fig.update_layout(
                            title=f"Source Localization - {method.upper()} (t = {results['times'][peak_time_idx]:.3f}s)",
                            scene=dict(
                                xaxis_title="X",
                                yaxis_title="Y",
                                zaxis_title="Z"
                            ),
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Time series of source activity
                        fig_time = go.Figure()
                        fig_time.add_trace(go.Scatter(
                            x=results['times'],
                            y=np.max(np.abs(results['data']), axis=0),
                            mode='lines',
                            name='Max Source Activity'
                        ))
                        
                        fig_time.update_layout(
                            title="Source Activity Over Time",
                            xaxis_title="Time (s)",
                            yaxis_title="Source Activity",
                            height=400
                        )
                        
                        st.plotly_chart(fig_time, use_container_width=True)
    
    elif analysis_type == "Connectivity Analysis":
        st.markdown("### 🌐 Connectivity Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            method = st.selectbox(
                "Method:",
                ["coh", "plv", "granger"],
                key="connectivity_method"
            )
        
        with col2:
            freq_min = st.number_input("Min Frequency (Hz)", value=6.0, key="conn_freq_min")
            freq_max = st.number_input("Max Frequency (Hz)", value=35.0, key="conn_freq_max")
        
        if st.button("Compute Connectivity"):
            with st.spinner("Computing connectivity..."):
                freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), 20)
                success = analyzer.compute_connectivity_analysis(epochs, method, freqs)
                
                if success:
                    st.success("Connectivity analysis completed!")
                    
                    # Display results
                    if analyzer.connectivity_results:
                        results = analyzer.connectivity_results
                        
                        # Create connectivity heatmap
                        fig = go.Figure()
                        
                        # Average across frequencies
                        avg_connectivity = np.mean(results['matrix'], axis=2)
                        
                        fig.add_trace(go.Heatmap(
                            z=avg_connectivity,
                            x=results['channels'],
                            y=results['channels'],
                            colorscale='Viridis',
                            colorbar=dict(title=f"{method.upper()} Connectivity")
                        ))
                        
                        fig.update_layout(
                            title=f"Average {method.upper()} Connectivity Matrix",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Frequency-dependent connectivity
                        fig_freq = go.Figure()
                        
                        # Plot connectivity for a few channel pairs
                        n_channels = len(results['channels'])
                        for i in range(min(5, n_channels)):
                            for j in range(i+1, min(i+3, n_channels)):
                                fig_freq.add_trace(go.Scatter(
                                    x=results['frequencies'],
                                    y=results['matrix'][i, j, :],
                                    mode='lines',
                                    name=f"{results['channels'][i]}-{results['channels'][j]}"
                                ))
                        
                        fig_freq.update_layout(
                            title="Frequency-Dependent Connectivity",
                            xaxis_title="Frequency (Hz)",
                            yaxis_title=f"{method.upper()} Connectivity",
                            height=400
                        )
                        
                        st.plotly_chart(fig_freq, use_container_width=True)
    
    elif analysis_type == "Non-linear Dynamics":
        st.markdown("### 🔄 Non-linear Dynamics")
        
        methods = st.multiselect(
            "Select Methods:",
            ["entropy", "lyapunov", "fractal"],
            default=["entropy"],
            key="nonlinear_methods"
        )
        
        if st.button("Compute Non-linear Measures"):
            with st.spinner("Computing non-linear measures..."):
                success = analyzer.compute_nonlinear_dynamics(epochs, methods)
                
                if success:
                    st.success("Non-linear analysis completed!")
                    
                    # Display results
                    if analyzer.nonlinear_results:
                        results = analyzer.nonlinear_results
                        
                        # Create subplots for each measure
                        n_measures = len(results)
                        fig = make_subplots(
                            rows=1, cols=n_measures,
                            subplot_titles=list(results.keys()),
                            specs=[[{"secondary_y": False}] * n_measures]
                        )
                        
                        for i, (measure, data) in enumerate(results.items()):
                            fig.add_trace(
                                go.Bar(
                                    x=epochs.ch_names,
                                    y=data['mean'],
                                    error_y=dict(type='data', array=data['std']),
                                    name=measure.capitalize()
                                ),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical summary
                        st.markdown("### 📊 Statistical Summary")
                        
                        summary_data = []
                        for measure, data in results.items():
                            summary_data.append({
                                'Measure': measure.capitalize(),
                                'Mean': f"{np.mean(data['mean']):.3f}",
                                'Std': f"{np.std(data['mean']):.3f}",
                                'Min': f"{np.min(data['mean']):.3f}",
                                'Max': f"{np.max(data['mean']):.3f}"
                            })
                        
                        import pandas as pd
                        df = pd.DataFrame(summary_data)
                        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    # Test the advanced analyzer
    analyzer = AdvancedAnalyzer()
    print("Advanced Analyzer initialized successfully!") 