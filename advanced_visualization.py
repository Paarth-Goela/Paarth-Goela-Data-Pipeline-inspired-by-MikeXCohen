"""
Advanced Visualization Suite
3D brain topography, interactive plots, and statistical overlays
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mne
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizer:
    """
    Advanced visualization tools for EEG analysis
    """
    
    def __init__(self):
        # Standard 10-20 electrode positions
        self.electrode_positions = {
            'Fp1': [-0.3, 0.9, 0.3], 'Fp2': [0.3, 0.9, 0.3],
            'F7': [-0.7, 0.7, 0.2], 'F3': [-0.5, 0.5, 0.7], 'Fz': [0, 0.5, 0.9], 'F4': [0.5, 0.5, 0.7], 'F8': [0.7, 0.7, 0.2],
            'FC5': [-0.6, 0.3, 0.7], 'FC1': [-0.2, 0.3, 0.9], 'FC2': [0.2, 0.3, 0.9], 'FC6': [0.6, 0.3, 0.7],
            'T7': [-0.8, 0, 0.6], 'C3': [-0.5, 0, 0.9], 'Cz': [0, 0, 1], 'C4': [0.5, 0, 0.9], 'T8': [0.8, 0, 0.6],
            'CP5': [-0.6, -0.3, 0.7], 'CP1': [-0.2, -0.3, 0.9], 'CP2': [0.2, -0.3, 0.9], 'CP6': [0.6, -0.3, 0.7],
            'P7': [-0.7, -0.7, 0.2], 'P3': [-0.5, -0.5, 0.7], 'Pz': [0, -0.5, 0.9], 'P4': [0.5, -0.5, 0.7], 'P8': [0.7, -0.7, 0.2],
            'PO9': [-0.3, -0.9, 0.3], 'O1': [-0.2, -0.9, 0.4], 'Oz': [0, -0.9, 0.4], 'O2': [0.2, -0.9, 0.4], 'PO10': [0.3, -0.9, 0.3],
            'AF7': [-0.5, 0.8, 0.3], 'AF3': [-0.3, 0.8, 0.5], 'AF4': [0.3, 0.8, 0.5], 'AF8': [0.5, 0.8, 0.3],
            'F5': [-0.6, 0.6, 0.5], 'F1': [-0.2, 0.6, 0.8], 'F2': [0.2, 0.6, 0.8], 'F6': [0.6, 0.6, 0.5],
            'FT9': [-0.9, 0.4, 0.2], 'FT7': [-0.8, 0.4, 0.4], 'FC3': [-0.4, 0.4, 0.8], 'FC4': [0.4, 0.4, 0.8], 'FT8': [0.8, 0.4, 0.4], 'FT10': [0.9, 0.4, 0.2],
            'C5': [-0.7, 0, 0.7], 'C1': [-0.2, 0, 1], 'C2': [0.2, 0, 1], 'C6': [0.7, 0, 0.7],
            'TP9': [-0.9, -0.4, 0.2], 'TP7': [-0.8, -0.4, 0.4], 'CP3': [-0.4, -0.4, 0.8], 'CP4': [0.4, -0.4, 0.8], 'TP8': [0.8, -0.4, 0.4], 'TP10': [0.9, -0.4, 0.2],
            'P5': [-0.6, -0.6, 0.5], 'P1': [-0.2, -0.6, 0.8], 'P2': [0.2, -0.6, 0.8], 'P6': [0.6, -0.6, 0.5],
            'PO7': [-0.5, -0.8, 0.3], 'PO3': [-0.3, -0.8, 0.5], 'PO4': [0.3, -0.8, 0.5], 'PO8': [0.5, -0.8, 0.3]
        }
    
    def create_3d_brain_topography(self, values, channel_names, title="3D Brain Topography", theme='dark'):
        """
        Create interactive 3D brain topography
        """
        # Theme configuration
        if theme == 'dark':
            template = 'plotly_dark'
            bg_color = '#0e1117'
            text_color = '#fafafa'
        else:
            template = 'plotly_white'
            bg_color = '#ffffff'
            text_color = '#262730'
        
        # Get electrode positions
        positions = []
        valid_values = []
        valid_channels = []
        
        for ch, val in zip(channel_names, values):
            if ch in self.electrode_positions:
                positions.append(self.electrode_positions[ch])
                valid_values.append(val)
                valid_channels.append(ch)
        
        if not positions:
            return None
        
        positions = np.array(positions)
        valid_values = np.array(valid_values)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add electrode points
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=valid_values,
                colorscale='Viridis',
                colorbar=dict(title="Value"),
                showscale=True
            ),
            text=valid_channels,
            textposition="middle center",
            name="Electrodes"
        ))
        
        # Add brain surface (simplified sphere)
        phi = np.linspace(0, 2*np.pi, 50)
        theta = np.linspace(0, np.pi, 25)
        phi, theta = np.meshgrid(phi, theta)
        
        # Create brain surface
        x = 1.2 * np.sin(theta) * np.cos(phi)
        y = 1.2 * np.sin(theta) * np.sin(phi)
        z = 1.2 * np.cos(theta)
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.1,
            colorscale='Greys',
            showscale=False,
            name="Brain Surface"
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            template=template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            height=600
        )
        
        return fig
    
    def create_connectivity_network(self, connectivity_matrix, channel_names, threshold=0.5, theme='dark'):
        """
        Create interactive connectivity network graph
        """
        # Theme configuration
        if theme == 'dark':
            template = 'plotly_dark'
            bg_color = '#0e1117'
            text_color = '#fafafa'
        else:
            template = 'plotly_white'
            bg_color = '#ffffff'
            text_color = '#262730'
        
        # Get electrode positions for layout
        positions = []
        valid_channels = []
        
        for ch in channel_names:
            if ch in self.electrode_positions:
                positions.append(self.electrode_positions[ch])
                valid_channels.append(ch)
        
        if not positions:
            return None
        
        positions = np.array(positions)
        
        # Create network edges
        edges_x = []
        edges_y = []
        edges_z = []
        edge_weights = []
        
        for i in range(len(valid_channels)):
            for j in range(i+1, len(valid_channels)):
                weight = connectivity_matrix[i, j]
                if weight > threshold:
                    # Add edge
                    edges_x.extend([positions[i, 0], positions[j, 0], None])
                    edges_y.extend([positions[i, 1], positions[j, 1], None])
                    edges_z.extend([positions[i, 2], positions[j, 2], None])
                    edge_weights.extend([weight, weight, None])
        
        # Create 3D network plot
        fig = go.Figure()
        
        # Add edges
        if edges_x:
            fig.add_trace(go.Scatter3d(
                x=edges_x,
                y=edges_y,
                z=edges_z,
                mode='lines',
                line=dict(
                    color=edge_weights,
                    colorscale='Viridis',
                    width=2
                ),
                opacity=0.6,
                name="Connections",
                showlegend=False
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            marker=dict(
                size=10,
                color='red',
                opacity=0.8
            ),
            text=valid_channels,
            textposition="middle center",
            name="Channels"
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Connectivity Network (threshold: {threshold})",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            template=template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            height=600
        )
        
        return fig
    
    def create_statistical_overlay(self, data, p_values, channel_names, time_points, 
                                 significance_level=0.05, theme='dark'):
        """
        Create time-frequency plot with statistical significance overlay
        """
        # Theme configuration
        if theme == 'dark':
            template = 'plotly_dark'
            bg_color = '#0e1117'
            text_color = '#fafafa'
        else:
            template = 'plotly_white'
            bg_color = '#ffffff'
            text_color = '#262730'
        
        # Create significance mask
        sig_mask = p_values < significance_level
        
        # Create main heatmap
        fig = go.Figure()
        
        # Add main data
        fig.add_trace(go.Heatmap(
            z=data,
            x=time_points,
            y=channel_names,
            colorscale='Viridis',
            name='Data',
            colorbar=dict(title="Value")
        ))
        
        # Add significance overlay
        if np.any(sig_mask):
            # Create significance overlay
            sig_data = np.where(sig_mask, data, np.nan)
            
            fig.add_trace(go.Heatmap(
                z=sig_data,
                x=time_points,
                y=channel_names,
                colorscale='Reds',
                opacity=0.7,
                name='Significant',
                showscale=False
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Statistical Overlay (p < {significance_level})",
            xaxis_title="Time (s)",
            yaxis_title="Channels",
            template=template,
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            font=dict(color=text_color),
            height=500
        )
        
        return fig
    
    def create_multi_subject_dashboard(self, subjects_data, theme='dark'):
        """
        Create multi-subject comparison dashboard
        """
        # Theme configuration
        if theme == 'dark':
            template = 'plotly_dark'
            bg_color = '#0e1117'
            text_color = '#fafafa'
        else:
            template = 'plotly_white'
            bg_color = '#ffffff'
            text_color = '#262730'
        
        n_subjects = len(subjects_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Subject Comparison', 'Group Statistics',
                          'Individual Traces', 'Correlation Matrix'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Subject comparison (box plot)
        subject_names = list(subjects_data.keys())
        subject_values = list(subjects_data.values())
        
        fig.add_trace(
            go.Box(
                y=subject_values,
                name='Subjects',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=1, col=1
        )
        
        # 2. Group statistics
        mean_values = [np.mean(vals) for vals in subject_values]
        std_values = [np.std(vals) for vals in subject_values]
        
        fig.add_trace(
            go.Bar(
                x=subject_names,
                y=mean_values,
                error_y=dict(type='data', array=std_values),
                name='Mean ± SD'
            ),
            row=1, col=2
        )
        
        # 3. Individual traces
        for i, (name, values) in enumerate(subjects_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(len(values)),
                    y=values,
                    name=f'Subject {name}',
                    mode='lines',
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # 4. Correlation matrix
        if n_subjects > 1:
            # Create correlation matrix
            corr_matrix = np.corrcoef(subject_values)
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=subject_names,
                    y=subject_names,
                    colorscale='RdBu_r',
                    zmid=0
                ),
                row=2, col=2
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
    
    def create_time_frequency_movie(self, tfr_data, freqs, times, channel_names, 
                                  output_path='tfr_movie.gif', fps=10, theme='dark'):
        """
        Create time-frequency movie with customizable parameters
        """
        import imageio
        
        # Theme configuration
        if theme == 'dark':
            plt.style.use('dark_background')
            bg_color = '#0e1117'
            text_color = '#fafafa'
        else:
            plt.style.use('default')
            bg_color = '#ffffff'
            text_color = '#262730'
        
        # Create frames
        frames = []
        n_time_points = len(times)
        
        for t_idx in range(0, n_time_points, max(1, n_time_points // (fps * 5))):  # 5 second movie
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot time-frequency data at this time point
            if tfr_data.ndim == 3:  # [channels, freqs, times]
                data_slice = tfr_data[:, :, t_idx]
            else:  # [freqs, times]
                data_slice = tfr_data[:, t_idx]
            
            # Create heatmap
            im = ax.imshow(data_slice, aspect='auto', origin='lower',
                          extent=[0, len(channel_names), freqs[0], freqs[-1]],
                          cmap='viridis')
            
            ax.set_xlabel('Channels')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'Time-Frequency Analysis - t = {times[t_idx]:.2f}s')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Power')
            
            # Save frame
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                       facecolor=bg_color)
            buf.seek(0)
            frame = imageio.imread(buf)
            frames.append(frame)
            plt.close()
        
        # Save movie
        imageio.mimsave(output_path, frames, duration=1/fps)
        
        return output_path

def create_advanced_visualization_interface():
    """Create Streamlit interface for advanced visualizations"""
    st.markdown("## 🎨 Advanced Visualization Suite")
    
    # Initialize visualizer
    if 'advanced_visualizer' not in st.session_state:
        st.session_state.advanced_visualizer = AdvancedVisualizer()
    
    visualizer = st.session_state.advanced_visualizer
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type:",
        ["3D Brain Topography", "Connectivity Network", "Statistical Overlay", 
         "Multi-Subject Dashboard", "Time-Frequency Movie"]
    )
    
    if viz_type == "3D Brain Topography":
        st.markdown("### 🧠 3D Brain Topography")
        
        # Generate sample data
        channel_names = list(visualizer.electrode_positions.keys())[:20]
        values = np.random.randn(len(channel_names))
        
        # Create 3D topography
        fig = visualizer.create_3d_brain_topography(
            values, channel_names, 
            theme=st.session_state.get('theme', 'dark')
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Connectivity Network":
        st.markdown("### 🌐 Connectivity Network")
        
        # Generate sample connectivity matrix
        n_channels = 20
        connectivity = np.random.rand(n_channels, n_channels)
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(connectivity, 0)  # No self-connections
        
        channel_names = list(visualizer.electrode_positions.keys())[:n_channels]
        
        threshold = st.slider("Connection Threshold", 0.0, 1.0, 0.5, 0.1)
        
        fig = visualizer.create_connectivity_network(
            connectivity, channel_names, threshold,
            theme=st.session_state.get('theme', 'dark')
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Statistical Overlay":
        st.markdown("### 📊 Statistical Overlay")
        
        # Generate sample data
        n_channels = 10
        n_timepoints = 100
        data = np.random.randn(n_channels, n_timepoints)
        p_values = np.random.random((n_channels, n_timepoints))
        time_points = np.linspace(0, 2, n_timepoints)
        channel_names = [f'Ch{i+1}' for i in range(n_channels)]
        
        significance_level = st.slider("Significance Level", 0.01, 0.1, 0.05, 0.01)
        
        fig = visualizer.create_statistical_overlay(
            data, p_values, channel_names, time_points, significance_level,
            theme=st.session_state.get('theme', 'dark')
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Multi-Subject Dashboard":
        st.markdown("### 👥 Multi-Subject Dashboard")
        
        # Generate sample multi-subject data
        subjects_data = {
            'S1': np.random.randn(50),
            'S2': np.random.randn(50),
            'S3': np.random.randn(50),
            'S4': np.random.randn(50),
            'S5': np.random.randn(50)
        }
        
        fig = visualizer.create_multi_subject_dashboard(
            subjects_data,
            theme=st.session_state.get('theme', 'dark')
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Time-Frequency Movie":
        st.markdown("### 🎬 Time-Frequency Movie")
        
        if st.button("Generate Time-Frequency Movie"):
            # Generate sample TFR data
            n_channels = 10
            n_freqs = 20
            n_times = 100
            
            tfr_data = np.random.randn(n_channels, n_freqs, n_times)
            freqs = np.linspace(1, 40, n_freqs)
            times = np.linspace(0, 2, n_times)
            channel_names = [f'Ch{i+1}' for i in range(n_channels)]
            
            # Create movie
            output_path = visualizer.create_time_frequency_movie(
                tfr_data, freqs, times, channel_names,
                theme=st.session_state.get('theme', 'dark')
            )
            
            # Display movie
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="Download Time-Frequency Movie",
                    data=f.read(),
                    file_name="tfr_movie.gif",
                    mime="image/gif"
                )
            
            st.success("Time-frequency movie generated!")

if __name__ == "__main__":
    # Test the visualizer
    visualizer = AdvancedVisualizer()
    
    # Test 3D topography
    channel_names = list(visualizer.electrode_positions.keys())[:20]
    values = np.random.randn(len(channel_names))
    
    fig = visualizer.create_3d_brain_topography(values, channel_names)
    if fig:
        fig.show() 