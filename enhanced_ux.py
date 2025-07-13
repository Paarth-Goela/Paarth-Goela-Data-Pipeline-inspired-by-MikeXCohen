"""
Enhanced UX Module for Neural Data Analysis Pipeline
Provides keyboard shortcuts, customizable dashboards, and advanced UI features
"""

import streamlit as st
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import keyboard
import threading
import time
from datetime import datetime

@dataclass
class KeyboardShortcut:
    """Keyboard shortcut configuration"""
    key: str
    description: str
    action: str
    category: str
    enabled: bool = True

@dataclass
class DashboardLayout:
    """Dashboard layout configuration"""
    name: str
    description: str
    layout_type: str  # 'grid', 'tabs', 'sidebar'
    components: List[Dict[str, Any]]
    theme: str = 'light'
    enabled: bool = True

class KeyboardShortcutManager:
    """Manage keyboard shortcuts for the application"""
    
    def __init__(self):
        self.shortcuts = self._get_default_shortcuts()
        self.active_shortcuts = {}
        self.shortcut_thread = None
        self.is_listening = False
        
    def _get_default_shortcuts(self) -> List[KeyboardShortcut]:
        """Get default keyboard shortcuts"""
        return [
            KeyboardShortcut('ctrl+shift+u', 'Upload Data', 'upload_data', 'Data Management'),
            KeyboardShortcut('ctrl+shift+p', 'Preprocess Data', 'preprocess_data', 'Data Processing'),
            KeyboardShortcut('ctrl+shift+a', 'Run Analysis', 'run_analysis', 'Analysis'),
            KeyboardShortcut('ctrl+shift+v', 'Advanced Visualization', 'advanced_viz', 'Visualization'),
            KeyboardShortcut('ctrl+shift+r', 'Real-time EEG', 'realtime_eeg', 'Real-time'),
            KeyboardShortcut('ctrl+shift+w', 'Workflow Manager', 'workflow_manager', 'Workflow'),
            KeyboardShortcut('ctrl+shift+d', 'Dark Mode Toggle', 'toggle_theme', 'UI'),
            KeyboardShortcut('ctrl+shift+s', 'Save Session', 'save_session', 'Session'),
            KeyboardShortcut('ctrl+shift+l', 'Load Session', 'load_session', 'Session'),
            KeyboardShortcut('ctrl+shift+e', 'Export Results', 'export_results', 'Export'),
            KeyboardShortcut('ctrl+shift+h', 'Show Help', 'show_help', 'Help'),
            KeyboardShortcut('ctrl+shift+c', 'Clear All', 'clear_all', 'Data Management'),
            KeyboardShortcut('f1', 'Quick Help', 'quick_help', 'Help'),
            KeyboardShortcut('f2', 'Toggle Sidebar', 'toggle_sidebar', 'UI'),
            KeyboardShortcut('f3', 'Fullscreen Mode', 'fullscreen_mode', 'UI'),
            KeyboardShortcut('f4', 'Reset View', 'reset_view', 'UI'),
            KeyboardShortcut('f5', 'Refresh Data', 'refresh_data', 'Data Management'),
            KeyboardShortcut('f6', 'Next Analysis', 'next_analysis', 'Analysis'),
            KeyboardShortcut('f7', 'Previous Analysis', 'prev_analysis', 'Analysis'),
            KeyboardShortcut('f8', 'Toggle Auto-save', 'toggle_autosave', 'Session'),
            KeyboardShortcut('f9', 'Performance Mode', 'performance_mode', 'UI'),
            KeyboardShortcut('f10', 'Debug Mode', 'debug_mode', 'Development'),
            KeyboardShortcut('f11', 'Screenshot', 'screenshot', 'Export'),
            KeyboardShortcut('f12', 'Developer Tools', 'dev_tools', 'Development'),
        ]
    
    def start_listening(self):
        """Start listening for keyboard shortcuts"""
        if not self.is_listening:
            self.is_listening = True
            self.shortcut_thread = threading.Thread(target=self._listen_for_shortcuts)
            self.shortcut_thread.daemon = True
            self.shortcut_thread.start()
    
    def stop_listening(self):
        """Stop listening for keyboard shortcuts"""
        self.is_listening = False
        if self.shortcut_thread:
            self.shortcut_thread.join()
    
    def _listen_for_shortcuts(self):
        """Listen for keyboard shortcuts in background thread"""
        while self.is_listening:
            for shortcut in self.shortcuts:
                if shortcut.enabled and keyboard.is_pressed(shortcut.key):
                    self._handle_shortcut(shortcut)
                    time.sleep(0.1)  # Prevent multiple triggers
            time.sleep(0.01)
    
    def _handle_shortcut(self, shortcut: KeyboardShortcut):
        """Handle keyboard shortcut action"""
        if shortcut.action in self.active_shortcuts:
            callback = self.active_shortcuts[shortcut.action]
            try:
                callback()
                st.success(f"Shortcut executed: {shortcut.description}")
            except Exception as e:
                st.error(f"Shortcut error: {e}")
    
    def register_shortcut(self, action: str, callback: Callable):
        """Register a callback for a shortcut action"""
        self.active_shortcuts[action] = callback
    
    def get_shortcuts_by_category(self) -> Dict[str, List[KeyboardShortcut]]:
        """Get shortcuts organized by category"""
        categories = {}
        for shortcut in self.shortcuts:
            if shortcut.category not in categories:
                categories[shortcut.category] = []
            categories[shortcut.category].append(shortcut)
        return categories
    
    def save_shortcuts_config(self, file_path: str):
        """Save shortcuts configuration to file"""
        config = {
            'shortcuts': [asdict(shortcut) for shortcut in self.shortcuts]
        }
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_shortcuts_config(self, file_path: str):
        """Load shortcuts configuration from file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            for shortcut_dict in config.get('shortcuts', []):
                for shortcut in self.shortcuts:
                    if shortcut.key == shortcut_dict['key']:
                        shortcut.enabled = shortcut_dict.get('enabled', True)
                        break

class DashboardManager:
    """Manage customizable dashboard layouts"""
    
    def __init__(self):
        self.layouts = self._get_default_layouts()
        self.current_layout = 'default'
        
    def _get_default_layouts(self) -> List[DashboardLayout]:
        """Get default dashboard layouts"""
        return [
            DashboardLayout(
                name='default',
                description='Default layout with sidebar and main area',
                layout_type='sidebar',
                components=[
                    {'type': 'sidebar', 'name': 'controls', 'position': 'left'},
                    {'type': 'main', 'name': 'visualization', 'position': 'center'},
                    {'type': 'sidebar', 'name': 'info', 'position': 'right'}
                ]
            ),
            DashboardLayout(
                name='analysis_focused',
                description='Layout optimized for analysis workflow',
                layout_type='tabs',
                components=[
                    {'type': 'tab', 'name': 'Data Upload', 'icon': '📁'},
                    {'type': 'tab', 'name': 'Preprocessing', 'icon': '⚙️'},
                    {'type': 'tab', 'name': 'Analysis', 'icon': '📊'},
                    {'type': 'tab', 'name': 'Results', 'icon': '📈'},
                    {'type': 'tab', 'name': 'Export', 'icon': '💾'}
                ]
            ),
            DashboardLayout(
                name='real_time',
                description='Layout for real-time data monitoring',
                layout_type='grid',
                components=[
                    {'type': 'grid', 'name': 'live_data', 'position': 'top', 'size': 'large'},
                    {'type': 'grid', 'name': 'controls', 'position': 'bottom_left', 'size': 'small'},
                    {'type': 'grid', 'name': 'metrics', 'position': 'bottom_right', 'size': 'small'}
                ]
            ),
            DashboardLayout(
                name='research',
                description='Layout for research and publication',
                layout_type='grid',
                components=[
                    {'type': 'grid', 'name': 'data_overview', 'position': 'top_left', 'size': 'medium'},
                    {'type': 'grid', 'name': 'statistics', 'position': 'top_right', 'size': 'medium'},
                    {'type': 'grid', 'name': 'plots', 'position': 'center', 'size': 'large'},
                    {'type': 'grid', 'name': 'results', 'position': 'bottom', 'size': 'large'}
                ]
            ),
            DashboardLayout(
                name='minimal',
                description='Minimal layout for focused work',
                layout_type='sidebar',
                components=[
                    {'type': 'sidebar', 'name': 'essential_controls', 'position': 'left'},
                    {'type': 'main', 'name': 'content', 'position': 'center'}
                ]
            )
        ]
    
    def get_layout(self, name: str) -> Optional[DashboardLayout]:
        """Get layout by name"""
        for layout in self.layouts:
            if layout.name == name:
                return layout
        return None
    
    def create_custom_layout(self, name: str, description: str, 
                           layout_type: str, components: List[Dict[str, Any]]) -> DashboardLayout:
        """Create a custom dashboard layout"""
        layout = DashboardLayout(
            name=name,
            description=description,
            layout_type=layout_type,
            components=components
        )
        self.layouts.append(layout)
        return layout
    
    def save_layouts_config(self, file_path: str):
        """Save layouts configuration to file"""
        config = {
            'layouts': [asdict(layout) for layout in self.layouts],
            'current_layout': self.current_layout
        }
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_layouts_config(self, file_path: str):
        """Load layouts configuration from file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            self.layouts = []
            for layout_dict in config.get('layouts', []):
                layout = DashboardLayout(**layout_dict)
                self.layouts.append(layout)
            
            self.current_layout = config.get('current_layout', 'default')

class SessionManager:
    """Manage user sessions and preferences"""
    
    def __init__(self, session_dir: str = "sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.current_session = None
        
    def save_session(self, session_name: str, data: Dict[str, Any]):
        """Save current session"""
        session_file = self.session_dir / f"{session_name}.json"
        
        session_data = {
            'name': session_name,
            'timestamp': time.time(),
            'data': data
        }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        st.success(f"Session saved: {session_name}")
    
    def load_session(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Load session by name"""
        session_file = self.session_dir / f"{session_name}.json"
        
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            self.current_session = session_name
            st.success(f"Session loaded: {session_name}")
            return session_data.get('data', {})
        
        st.error(f"Session not found: {session_name}")
        return None
    
    def list_sessions(self) -> List[str]:
        """List available sessions"""
        session_files = list(self.session_dir.glob("*.json"))
        return [f.stem for f in session_files]
    
    def delete_session(self, session_name: str):
        """Delete a session"""
        session_file = self.session_dir / f"{session_name}.json"
        
        if session_file.exists():
            session_file.unlink()
            st.success(f"Session deleted: {session_name}")
        else:
            st.error(f"Session not found: {session_name}")
    
    def auto_save_session(self, data: Dict[str, Any]):
        """Auto-save session with timestamp"""
        if self.current_session:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            auto_save_name = f"{self.current_session}_autosave_{timestamp}"
            self.save_session(auto_save_name, data)

class PerformanceOptimizer:
    """Optimize application performance"""
    
    def __init__(self):
        self.performance_mode = False
        self.cache_enabled = True
        self.auto_refresh = False
        
    def enable_performance_mode(self):
        """Enable performance optimization mode"""
        self.performance_mode = True
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Reduce plot complexity
        st.session_state['plot_quality'] = 'low'
        st.session_state['auto_refresh'] = False
        
        st.success("Performance mode enabled")
    
    def disable_performance_mode(self):
        """Disable performance optimization mode"""
        self.performance_mode = False
        st.session_state['plot_quality'] = 'high'
        st.session_state['auto_refresh'] = True
        
        st.success("Performance mode disabled")
    
    def optimize_plot_settings(self, plot_type: str) -> Dict[str, Any]:
        """Get optimized plot settings based on performance mode"""
        if self.performance_mode:
            return {
                'points': 1000,  # Reduce data points
                'quality': 'low',
                'auto_refresh': False,
                'animation': False
            }
        else:
            return {
                'points': 10000,  # Full data points
                'quality': 'high',
                'auto_refresh': True,
                'animation': True
            }

class AccessibilityManager:
    """Manage accessibility features"""
    
    def __init__(self):
        self.high_contrast = False
        self.large_fonts = False
        self.screen_reader = False
        self.keyboard_navigation = True
        
    def enable_high_contrast(self):
        """Enable high contrast mode"""
        self.high_contrast = True
        st.markdown("""
        <style>
        .stApp {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .stButton > button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #ffffff !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.success("High contrast mode enabled")
    
    def disable_high_contrast(self):
        """Disable high contrast mode"""
        self.high_contrast = False
        st.success("High contrast mode disabled")
    
    def enable_large_fonts(self):
        """Enable large fonts"""
        self.large_fonts = True
        st.markdown("""
        <style>
        .stMarkdown, .stText, .stSelectbox, .stSlider {
            font-size: 18px !important;
        }
        .stButton > button {
            font-size: 18px !important;
            padding: 12px 24px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.success("Large fonts enabled")
    
    def disable_large_fonts(self):
        """Disable large fonts"""
        self.large_fonts = False
        st.success("Large fonts disabled")

def render_enhanced_sidebar():
    """Render enhanced sidebar with advanced UX features"""
    
    with st.sidebar:
        st.header("🎛️ Enhanced Controls")
        
        # Keyboard Shortcuts Section
        with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
            shortcut_manager = KeyboardShortcutManager()
            categories = shortcut_manager.get_shortcuts_by_category()
            
            for category, shortcuts in categories.items():
                st.subheader(category)
                for shortcut in shortcuts:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.code(shortcut.key, language=None)
                    with col2:
                        st.write(f"{shortcut.description}")
                        if not shortcut.enabled:
                            st.caption("(Disabled)")
        
        # Dashboard Layouts Section
        with st.expander("📊 Dashboard Layouts", expanded=False):
            dashboard_manager = DashboardManager()
            
            layout_names = [layout.name for layout in dashboard_manager.layouts]
            selected_layout = st.selectbox(
                "Choose Layout",
                layout_names,
                index=0
            )
            
            if st.button("Apply Layout"):
                st.session_state['current_layout'] = selected_layout
                st.success(f"Layout applied: {selected_layout}")
            
            # Layout descriptions
            layout = dashboard_manager.get_layout(selected_layout)
            if layout:
                st.caption(layout.description)
        
        # Session Management Section
        with st.expander("💾 Session Management", expanded=False):
            session_manager = SessionManager()
            
            # Save session
            session_name = st.text_input("Session Name", value="my_session")
            if st.button("Save Session"):
                session_data = {
                    'uploaded_files': st.session_state.get('uploaded_files', []),
                    'analysis_results': st.session_state.get('analysis_results', {}),
                    'current_page': st.session_state.get('current_page', 'main'),
                    'theme': st.session_state.get('theme', 'light')
                }
                session_manager.save_session(session_name, session_data)
            
            # Load session
            sessions = session_manager.list_sessions()
            if sessions:
                selected_session = st.selectbox("Load Session", sessions)
                if st.button("Load Selected Session"):
                    session_data = session_manager.load_session(selected_session)
                    if session_data:
                        # Restore session state
                        for key, value in session_data.items():
                            st.session_state[key] = value
        
        # Performance Settings Section
        with st.expander("⚡ Performance", expanded=False):
            performance_optimizer = PerformanceOptimizer()
            
            if st.button("Enable Performance Mode"):
                performance_optimizer.enable_performance_mode()
            
            if st.button("Disable Performance Mode"):
                performance_optimizer.disable_performance_mode()
            
            st.checkbox("Enable Caching", value=True, key="cache_enabled")
            st.checkbox("Auto-refresh", value=False, key="auto_refresh")
        
        # Accessibility Section
        with st.expander("♿ Accessibility", expanded=False):
            accessibility_manager = AccessibilityManager()
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("High Contrast"):
                    accessibility_manager.enable_high_contrast()
                if st.button("Large Fonts"):
                    accessibility_manager.enable_large_fonts()
            
            with col2:
                if st.button("Normal Contrast"):
                    accessibility_manager.disable_high_contrast()
                if st.button("Normal Fonts"):
                    accessibility_manager.disable_large_fonts()
        
        # Quick Actions Section
        with st.expander("⚡ Quick Actions", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📁 Upload", use_container_width=True):
                    st.session_state['current_page'] = 'main'
                
                if st.button("📊 Analysis", use_container_width=True):
                    st.session_state['current_page'] = 'advanced_analysis'
                
                if st.button("📈 Visualization", use_container_width=True):
                    st.session_state['current_page'] = 'advanced_visualization'
            
            with col2:
                if st.button("⚡ Real-time", use_container_width=True):
                    st.session_state['current_page'] = 'realtime_eeg'
                
                if st.button("🔄 Workflow", use_container_width=True):
                    st.session_state['current_page'] = 'workflow'
                
                if st.button("💾 Export", use_container_width=True):
                    st.session_state['current_page'] = 'export'

def render_keyboard_shortcuts_help():
    """Render keyboard shortcuts help page"""
    st.header("⌨️ Keyboard Shortcuts")
    
    shortcut_manager = KeyboardShortcutManager()
    categories = shortcut_manager.get_shortcuts_by_category()
    
    for category, shortcuts in categories.items():
        st.subheader(category)
        
        # Create a table-like display
        for shortcut in shortcuts:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.code(shortcut.key, language=None)
            with col2:
                st.write(shortcut.description)
            with col3:
                if shortcut.enabled:
                    st.success("✓")
                else:
                    st.error("✗")
        
        st.divider()

def render_dashboard_customization():
    """Render dashboard customization interface"""
    st.header("📊 Dashboard Customization")
    
    dashboard_manager = DashboardManager()
    
    # Current layout info
    current_layout = st.session_state.get('current_layout', 'default')
    layout = dashboard_manager.get_layout(current_layout)
    
    if layout:
        st.subheader(f"Current Layout: {layout.name}")
        st.write(layout.description)
        
        # Layout components
        st.write("**Components:**")
        for component in layout.components:
            st.write(f"- {component['type']}: {component['name']}")
    
    # Create custom layout
    st.subheader("Create Custom Layout")
    
    with st.form("custom_layout_form"):
        layout_name = st.text_input("Layout Name")
        layout_description = st.text_area("Description")
        layout_type = st.selectbox("Layout Type", ['sidebar', 'tabs', 'grid'])
        
        submitted = st.form_submit_button("Create Layout")
        
        if submitted and layout_name:
            # Simple component configuration
            components = [
                {'type': 'main', 'name': 'content', 'position': 'center'}
            ]
            
            custom_layout = dashboard_manager.create_custom_layout(
                layout_name, layout_description, layout_type, components
            )
            
            st.success(f"Custom layout '{layout_name}' created!")

def initialize_enhanced_ux():
    """Initialize enhanced UX features"""
    
    # Initialize session state for enhanced UX
    if 'current_layout' not in st.session_state:
        st.session_state['current_layout'] = 'default'
    
    if 'plot_quality' not in st.session_state:
        st.session_state['plot_quality'] = 'high'
    
    if 'auto_refresh' not in st.session_state:
        st.session_state['auto_refresh'] = True
    
    if 'cache_enabled' not in st.session_state:
        st.session_state['cache_enabled'] = True
    
    # Start keyboard shortcut listener
    shortcut_manager = KeyboardShortcutManager()
    shortcut_manager.start_listening()
    
    # Register default shortcuts
    def upload_data_shortcut():
        st.session_state['current_page'] = 'main'
    
    def analysis_shortcut():
        st.session_state['current_page'] = 'advanced_analysis'
    
    def visualization_shortcut():
        st.session_state['current_page'] = 'advanced_visualization'
    
    def realtime_shortcut():
        st.session_state['current_page'] = 'realtime_eeg'
    
    def workflow_shortcut():
        st.session_state['current_page'] = 'workflow'
    
    def toggle_theme_shortcut():
        current_theme = st.session_state.get('theme', 'light')
        st.session_state['theme'] = 'dark' if current_theme == 'light' else 'light'
    
    # Register shortcuts
    shortcut_manager.register_shortcut('upload_data', upload_data_shortcut)
    shortcut_manager.register_shortcut('run_analysis', analysis_shortcut)
    shortcut_manager.register_shortcut('advanced_viz', visualization_shortcut)
    shortcut_manager.register_shortcut('realtime_eeg', realtime_shortcut)
    shortcut_manager.register_shortcut('workflow_manager', workflow_shortcut)
    shortcut_manager.register_shortcut('toggle_theme', toggle_theme_shortcut)
    
    return shortcut_manager 