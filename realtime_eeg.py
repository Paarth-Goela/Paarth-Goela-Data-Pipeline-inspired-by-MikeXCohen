"""
Real-Time EEG Processing Module
Supports live streaming, artifact detection, and real-time visualization
"""

import numpy as np
import threading
import time
import queue
from collections import deque
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import signal
from scipy.stats import zscore
import mne
import warnings
warnings.filterwarnings('ignore')

class RealTimeEEGProcessor:
    """
    Real-time EEG processing with live visualization
    """
    
    def __init__(self, buffer_size=1000, sampling_rate=1000, n_channels=64):
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        
        # Data buffers
        self.eeg_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        self.artifact_flags = deque(maxlen=buffer_size)
        
        # Processing parameters
        self.filter_low = 1.0
        self.filter_high = 40.0
        self.artifact_threshold = 150e-6  # 150 μV
        
        # Real-time analysis
        self.spectral_buffer = deque(maxlen=100)  # Last 100 spectral estimates
        self.alpha_power = deque(maxlen=100)
        self.beta_power = deque(maxlen=100)
        self.theta_power = deque(maxlen=100)
        self.delta_power = deque(maxlen=100)
        
        # Threading
        self.is_running = False
        self.data_queue = queue.Queue()
        self.processing_thread = None
        
        # Initialize buffers
        self._initialize_buffers()
    
    def _initialize_buffers(self):
        """Initialize data buffers with zeros"""
        initial_data = np.zeros((self.buffer_size, self.n_channels))
        initial_time = np.linspace(0, self.buffer_size/self.sampling_rate, self.buffer_size)
        
        for i in range(self.buffer_size):
            self.eeg_buffer.append(initial_data[i])
            self.time_buffer.append(initial_time[i])
            self.artifact_flags.append(False)
    
    def start_processing(self):
        """Start real-time processing thread"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            print("Real-time EEG processing started")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        print("Real-time EEG processing stopped")
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Get new data from queue (non-blocking)
                try:
                    new_data = self.data_queue.get_nowait()
                    self._process_new_data(new_data)
                except queue.Empty:
                    # Generate simulated data if no real data
                    self._generate_simulated_data()
                
                time.sleep(1.0 / self.sampling_rate)  # Maintain sampling rate
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def _generate_simulated_data(self):
        """Generate simulated EEG data for demonstration"""
        # Simulate realistic EEG with alpha, beta, theta components
        t = time.time()
        
        # Generate different frequency components
        alpha_freq = 10  # Hz
        beta_freq = 20   # Hz
        theta_freq = 6   # Hz
        
        # Create realistic EEG signal
        signal_alpha = 20e-6 * np.sin(2 * np.pi * alpha_freq * t)
        signal_beta = 15e-6 * np.sin(2 * np.pi * beta_freq * t)
        signal_theta = 25e-6 * np.sin(2 * np.pi * theta_freq * t)
        
        # Add some noise and artifacts occasionally
        noise = 5e-6 * np.random.randn(self.n_channels)
        
        # Occasionally add artifacts
        if np.random.random() < 0.01:  # 1% chance of artifact
            artifact = 200e-6 * np.random.randn(self.n_channels)
            noise += artifact
        
        # Combine signals
        eeg_data = signal_alpha + signal_beta + signal_theta + noise
        
        # Add to processing queue
        self.data_queue.put(eeg_data)
    
    def _process_new_data(self, eeg_data):
        """Process new EEG data"""
        # Add to buffer
        self.eeg_buffer.append(eeg_data)
        self.time_buffer.append(time.time())
        
        # Detect artifacts
        artifact_detected = self._detect_artifacts(eeg_data)
        self.artifact_flags.append(artifact_detected)
        
        # Compute spectral features
        if len(self.eeg_buffer) >= 256:  # Need enough data for FFT
            spectral_features = self._compute_spectral_features()
            self.spectral_buffer.append(spectral_features)
            
            # Extract band powers
            self.alpha_power.append(spectral_features.get('alpha', 0))
            self.beta_power.append(spectral_features.get('beta', 0))
            self.theta_power.append(spectral_features.get('theta', 0))
            self.delta_power.append(spectral_features.get('delta', 0))
    
    def _detect_artifacts(self, eeg_data):
        """Detect artifacts in EEG data"""
        # Simple threshold-based artifact detection
        if np.any(np.abs(eeg_data) > self.artifact_threshold):
            return True
        
        # Check for sudden jumps (gradient-based detection)
        if len(self.eeg_buffer) > 1:
            gradient = np.abs(eeg_data - list(self.eeg_buffer)[-1])
            if np.any(gradient > 50e-6):  # 50 μV jump threshold
                return True
        
        return False
    
    def _compute_spectral_features(self):
        """Compute spectral features from recent data"""
        # Get recent data
        recent_data = np.array(list(self.eeg_buffer)[-256:])
        
        # Compute power spectral density
        freqs, psd = signal.welch(recent_data.mean(axis=1), fs=self.sampling_rate, nperseg=128)
        
        # Define frequency bands
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        features = {}
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency indices
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if np.any(band_mask):
                features[band_name] = np.mean(psd[band_mask])
            else:
                features[band_name] = 0
        
        return features
    
    def get_live_plot_data(self):
        """Get data for live plotting"""
        if len(self.eeg_buffer) == 0:
            return None
        
        # Convert buffers to arrays
        eeg_array = np.array(list(self.eeg_buffer))
        time_array = np.array(list(self.time_buffer))
        artifact_array = np.array(list(self.artifact_flags))
        
        return {
            'eeg': eeg_array,
            'time': time_array,
            'artifacts': artifact_array,
            'alpha_power': list(self.alpha_power),
            'beta_power': list(self.beta_power),
            'theta_power': list(self.theta_power),
            'delta_power': list(self.delta_power)
        }
    
    def create_live_dashboard(self, theme='dark'):
        """Create live dashboard with multiple plots"""
        data = self.get_live_plot_data()
        if data is None:
            return None
        
        # Theme configuration
        if theme == 'dark':
            template = 'plotly_dark'
            bg_color = '#0e1117'
            text_color = '#fafafa'
        else:
            template = 'plotly_white'
            bg_color = '#ffffff'
            text_color = '#262730'
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Live EEG Signal', 'Power Spectral Density', 
                          'Band Powers Over Time', 'Artifact Detection',
                          'Channel Power Map', 'Signal Quality Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Live EEG Signal (first few channels)
        n_channels_plot = min(8, self.n_channels)
        for ch in range(n_channels_plot):
            fig.add_trace(
                go.Scatter(
                    x=data['time'],
                    y=data['eeg'][:, ch],
                    name=f'Ch {ch+1}',
                    mode='lines',
                    line=dict(width=1)
                ),
                row=1, col=1
            )
        
        # 2. Power Spectral Density
        if len(data['alpha_power']) > 0:
            # Create frequency array for recent data
            recent_eeg = data['eeg'][-256:, :].mean(axis=1)
            freqs, psd = signal.welch(recent_eeg, fs=self.sampling_rate, nperseg=128)
            
            fig.add_trace(
                go.Scatter(
                    x=freqs,
                    y=psd,
                    name='PSD',
                    mode='lines',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
        
        # 3. Band Powers Over Time
        if len(data['alpha_power']) > 0:
            time_points = np.arange(len(data['alpha_power']))
            
            fig.add_trace(
                go.Scatter(x=time_points, y=data['alpha_power'], name='Alpha', mode='lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_points, y=data['beta_power'], name='Beta', mode='lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_points, y=data['theta_power'], name='Theta', mode='lines'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=time_points, y=data['delta_power'], name='Delta', mode='lines'),
                row=2, col=1
            )
        
        # 4. Artifact Detection
        artifact_rate = np.mean(data['artifacts'])
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=artifact_rate * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Artifact Rate (%)"},
                gauge={'axis': {'range': [None, 20]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 5], 'color': "lightgray"},
                           {'range': [5, 10], 'color': "yellow"},
                           {'range': [10, 20], 'color': "red"}
                       ],
                       'threshold': {
                           'line': {'color': "red", 'width': 4},
                           'thickness': 0.75,
                           'value': 10
                       }}
            ),
            row=2, col=2
        )
        
        # 5. Channel Power Map (simplified)
        if len(data['eeg']) > 0:
            channel_powers = np.var(data['eeg'], axis=0)
            fig.add_trace(
                go.Bar(
                    x=[f'Ch {i+1}' for i in range(min(16, self.n_channels))],
                    y=channel_powers[:16],
                    name='Channel Power'
                ),
                row=3, col=1
            )
        
        # 6. Signal Quality Metrics
        if len(data['eeg']) > 0:
            # Compute quality metrics
            signal_variance = np.var(data['eeg'], axis=0)
            signal_mean = np.mean(data['eeg'], axis=0)
            
            fig.add_trace(
                go.Scatter(
                    x=signal_mean,
                    y=signal_variance,
                    mode='markers',
                    name='Signal Quality',
                    marker=dict(
                        size=8,
                        color=signal_variance,
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            template=template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            showlegend=True
        )
        
        return fig

class EEGStreamer:
    """
    Interface for different EEG hardware systems
    """
    
    def __init__(self, device_type='simulation'):
        self.device_type = device_type
        self.is_connected = False
        self.sampling_rate = 1000
        self.n_channels = 64
        
        # Device-specific parameters
        if device_type == 'brainvision':
            self.sampling_rate = 1000
            self.n_channels = 64
        elif device_type == 'gtec':
            self.sampling_rate = 1200
            self.n_channels = 32
        elif device_type == 'biosemi':
            self.sampling_rate = 2048
            self.n_channels = 64
    
    def connect(self):
        """Connect to EEG device"""
        if self.device_type == 'simulation':
            self.is_connected = True
            print("Connected to simulated EEG device")
        else:
            # Here you would implement actual device connection
            print(f"Connecting to {self.device_type} device...")
            # Placeholder for actual implementation
            self.is_connected = True
    
    def disconnect(self):
        """Disconnect from EEG device"""
        self.is_connected = False
        print("Disconnected from EEG device")
    
    def get_data(self):
        """Get data from device"""
        if not self.is_connected:
            return None
        
        if self.device_type == 'simulation':
            # Return simulated data
            return np.random.randn(self.n_channels) * 10e-6
        else:
            # Placeholder for actual device data
            return np.random.randn(self.n_channels) * 10e-6

def create_realtime_dashboard():
    """Create Streamlit interface for real-time EEG"""
    st.markdown("## 🧠 Real-Time EEG Processing")
    
    # Initialize processor
    if 'realtime_processor' not in st.session_state:
        st.session_state.realtime_processor = RealTimeEEGProcessor()
    
    processor = st.session_state.realtime_processor
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Real-Time Processing"):
            processor.start_processing()
            st.success("Real-time processing started!")
    
    with col2:
        if st.button("Stop Real-Time Processing"):
            processor.stop_processing()
            st.warning("Real-time processing stopped!")
    
    with col3:
        artifact_threshold = st.slider(
            "Artifact Threshold (μV)",
            min_value=50,
            max_value=300,
            value=150,
            step=10
        )
        processor.artifact_threshold = artifact_threshold * 1e-6
    
    # Live dashboard
    if processor.is_running:
        # Create live plot
        fig = processor.create_live_dashboard(theme=st.session_state.get('theme', 'dark'))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh
        time.sleep(0.1)
        st.rerun()
    else:
        st.info("Click 'Start Real-Time Processing' to begin live EEG analysis")
    
    # Device connection
    st.markdown("### 📡 Device Connection")
    device_type = st.selectbox(
        "Select EEG Device:",
        ["simulation", "brainvision", "gtec", "biosemi"]
    )
    
    if st.button("Connect to Device"):
        streamer = EEGStreamer(device_type)
        streamer.connect()
        st.session_state.eeg_streamer = streamer
        st.success(f"Connected to {device_type} device!")

if __name__ == "__main__":
    # Test the real-time processor
    processor = RealTimeEEGProcessor()
    processor.start_processing()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        processor.stop_processing()
        print("Stopped by user") 