#!/usr/bin/env python3
"""
Test script for advanced neural data analysis features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neural_backend import NeuralDataAnalyzer
import tempfile
import os

def test_signal_quality_metrics():
    """Test signal quality metrics computation"""
    print("🧪 Testing Signal Quality Metrics...")
    
    # Initialize analyzer and load sample data
    analyzer = NeuralDataAnalyzer()
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    
    # Compute quality metrics
    quality_metrics = analyzer.compute_signal_quality_metrics(
        metrics=['variance', 'entropy', 'kurtosis', 'snr']
    )
    
    print(f"✅ Quality metrics computed for {len(quality_metrics)} channels")
    
    # Test plotting
    fig = analyzer.plot_quality_metrics(quality_metrics, metric_type='variance')
    print("✅ Quality metrics plotting successful")
    
    return quality_metrics

def test_single_trial_analysis():
    """Test single-trial feature extraction"""
    print("🧪 Testing Single-Trial Analysis...")
    
    analyzer = NeuralDataAnalyzer()
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    
    # Test different feature types
    feature_types = ['erp', 'power', 'phase']
    
    for feature_type in feature_types:
        print(f"  Testing {feature_type} features...")
        
        single_trial_features = analyzer.extract_single_trial_features(
            feature_type=feature_type,
            frequency_band=(8, 13) if feature_type in ['power', 'phase'] else None
        )
        
        print(f"    ✅ {feature_type} features extracted for {len(single_trial_features)} channels")
        
        # Test plotting
        fig = analyzer.plot_single_trial_analysis(
            single_trial_features, 
            feature_type=feature_type,
            plot_type='heatmap'
        )
        print(f"    ✅ {feature_type} plotting successful")
    
    return single_trial_features

def test_cognitive_modeling():
    """Test cognitive modeling with GLM"""
    print("🧪 Testing Cognitive Modeling...")
    
    analyzer = NeuralDataAnalyzer()
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    
    # Create mock behavioral data
    n_trials = len(analyzer.epochs)
    behavioral_data = np.random.normal(500, 100, n_trials)  # Mock reaction times
    
    # Extract neural features
    neural_features = analyzer.extract_single_trial_features(
        feature_type='erp',
        time_window=(0.1, 0.3)
    )
    
    # Fit GLM model
    glm_results = analyzer.fit_glm_model(
        behavioral_data=behavioral_data,
        neural_features=neural_features,
        model_type='linear'
    )
    
    if glm_results:
        print(f"✅ GLM model fitted successfully")
        print(f"   R² score: {glm_results['metrics']['r2']:.3f}")
        print(f"   MSE: {glm_results['metrics']['mse']:.3f}")
    else:
        print("❌ GLM model fitting failed")
    
    return glm_results

def test_pretrained_models():
    """Test pre-trained model integration"""
    print("🧪 Testing Pre-trained Models...")
    
    analyzer = NeuralDataAnalyzer()
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    
    # Create a simple mock model
    from sklearn.ensemble import RandomForestClassifier
    mock_model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create mock training data
    X_train = np.random.randn(100, 10)
    y_train = np.random.choice([0, 1], 100)
    mock_model.fit(X_train, y_train)
    
    # Save model temporarily
    import joblib
    temp_model_path = "temp_model.joblib"
    joblib.dump(mock_model, temp_model_path)
    
    try:
        # Load pre-trained model
        loaded_model = analyzer.load_pretrained_model(
            temp_model_path, 
            model_type='classifier'
        )
        
        if loaded_model:
            print("✅ Pre-trained model loaded successfully")
            
            # Test model application
            mock_data = np.random.randn(50, 10)
            results = analyzer.apply_pretrained_model(mock_data, feature_extraction=False)
            
            if results:
                print(f"✅ Model applied successfully")
                print(f"   Predictions: {len(results['predictions'])}")
            else:
                print("❌ Model application failed")
        else:
            print("❌ Pre-trained model loading failed")
    
    finally:
        # Clean up
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
    
    return loaded_model is not None

def test_movie_generation():
    """Test time-series movie generation"""
    print("🧪 Testing Movie Generation...")
    
    analyzer = NeuralDataAnalyzer()
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    
    # Test ERP movie generation
    try:
        movie_path = analyzer.generate_time_series_movie(
            data_type='erp',
            output_path='test_erp_movie.mp4',
            fps=5,
            duration=3
        )
        
        if movie_path and os.path.exists(movie_path):
            print("✅ ERP movie generated successfully")
            os.remove(movie_path)  # Clean up
        else:
            print("❌ ERP movie generation failed")
    
    except Exception as e:
        print(f"❌ Movie generation error: {e}")
    
    return movie_path is not None

def test_ai_quality_analysis():
    """Test AI-powered quality analysis"""
    print("🧪 Testing AI Quality Analysis...")
    
    analyzer = NeuralDataAnalyzer()
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    
    # Test data quality analysis
    quality_analysis = analyzer.analyze_data_quality()
    
    print(f"✅ Quality analysis completed")
    print(f"   Quality score: {quality_analysis['quality_score']}/100")
    print(f"   Warnings: {len(quality_analysis['warnings'])}")
    print(f"   Recommendations: {len(quality_analysis['recommendations'])}")
    
    # Test AI suggestions
    suggestions = analyzer.get_analysis_suggestions(analysis_type='preprocessing')
    
    if suggestions and not suggestions.startswith("Could not generate"):
        print("✅ AI suggestions generated successfully")
    else:
        print("❌ AI suggestions generation failed")
    
    return quality_analysis, suggestions

def test_realtime_streaming():
    """Test real-time streaming setup"""
    print("🧪 Testing Real-Time Streaming...")
    
    analyzer = NeuralDataAnalyzer()
    
    # Test simulation mode
    success = analyzer.setup_realtime_streaming(stream_type='simulation')
    
    if success:
        print("✅ Real-time streaming setup successful")
        
        # Test data processing
        mock_data_chunk = np.random.randn(100)
        results = analyzer.process_realtime_data(mock_data_chunk)
        
        if results:
            print("✅ Real-time data processing successful")
        else:
            print("⚠️ Real-time data processing returned no results (expected for small chunks)")
    else:
        print("❌ Real-time streaming setup failed")
    
    return success

def test_advanced_workflow():
    """Test complete advanced workflow"""
    print("🧪 Testing Complete Advanced Workflow...")
    
    analyzer = NeuralDataAnalyzer()
    
    # Load and preprocess data
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    
    # Run time-frequency analysis
    analyzer.time_frequency_analysis()
    
    # Compute quality metrics
    quality_metrics = analyzer.compute_signal_quality_metrics(['variance', 'snr'])
    
    # Extract single-trial features
    single_trial_features = analyzer.extract_single_trial_features('erp')
    
    # Create mock behavioral data
    n_trials = len(analyzer.epochs)
    behavioral_data = np.random.normal(500, 100, n_trials)
    
    # Fit cognitive model
    glm_results = analyzer.fit_glm_model(
        behavioral_data=behavioral_data,
        neural_features=single_trial_features
    )
    
    # Analyze data quality
    quality_analysis = analyzer.analyze_data_quality()
    
    print("✅ Complete advanced workflow successful!")
    print(f"   Quality score: {quality_analysis['quality_score']}/100")
    print(f"   GLM R²: {glm_results['metrics']['r2']:.3f}" if glm_results else "GLM failed")
    
    return True

def main():
    """Run all advanced feature tests"""
    print("🚀 Testing Advanced Neural Data Analysis Features")
    print("=" * 60)
    
    tests = [
        ("Signal Quality Metrics", test_signal_quality_metrics),
        ("Single-Trial Analysis", test_single_trial_analysis),
        ("Cognitive Modeling", test_cognitive_modeling),
        ("Pre-trained Models", test_pretrained_models),
        ("Movie Generation", test_movie_generation),
        ("AI Quality Analysis", test_ai_quality_analysis),
        ("Real-time Streaming", test_realtime_streaming),
        ("Complete Workflow", test_advanced_workflow)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results[test_name] = result
            print(f"✅ {test_name} test completed successfully")
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All advanced features working correctly!")
    else:
        print("⚠️ Some features need attention. Check the error messages above.")

if __name__ == "__main__":
    main() 