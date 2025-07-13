#!/usr/bin/env python3
"""
Test script for PAC functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from neural_backend import NeuralDataAnalyzer

def test_pac_functionality():
    """Test PAC computation and plotting"""
    print("🧠 Testing PAC Functionality")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = NeuralDataAnalyzer()
    
    try:
        # Load sample data
        print("📊 Loading sample data...")
        success = analyzer.load_sample_data()
        if not success:
            print("❌ Failed to load sample data")
            return False
        
        # Preprocess
        print("🔧 Preprocessing data...")
        analyzer.preprocess_data(l_freq=1.0, h_freq=40.0)
        
        # Create epochs
        print("⏱️  Creating epochs...")
        analyzer.create_epochs(tmin=-0.5, tmax=1.0)
        
        # Test PAC computation
        print("🔄 Computing PAC...")
        success = analyzer.compute_phase_amplitude_coupling(
            low_freq=(4, 8), 
            high_freq=(30, 40), 
            method='tort'
        )
        
        if not success:
            print("❌ PAC computation failed")
            return False
        
        print("✅ PAC computation successful!")
        
        # Test PAC plotting
        print("📈 Testing PAC plotting...")
        
        # Test histogram method
        try:
            fig_hist = analyzer.plot_pac_results(method='histogram')
            print("✅ Histogram plotting successful")
            plt.close(fig_hist)
        except Exception as e:
            print(f"❌ Histogram plotting failed: {e}")
        
        # Test boxplot method
        try:
            fig_box = analyzer.plot_pac_results(method='boxplot')
            print("✅ Boxplot plotting successful")
            plt.close(fig_box)
        except Exception as e:
            print(f"❌ Boxplot plotting failed: {e}")
        
        # Test summary method
        try:
            fig_summary = analyzer.plot_pac_results(method='summary')
            print("✅ Summary plotting successful")
            plt.close(fig_summary)
        except Exception as e:
            print(f"❌ Summary plotting failed: {e}")
        
        print("\n🎉 PAC functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_pac_functionality() 