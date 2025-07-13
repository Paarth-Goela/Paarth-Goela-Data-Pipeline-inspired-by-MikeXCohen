import os
from neural_backend import NeuralDataAnalyzer

def test_movie_generation(data_type, output_path):
    print(f"\nTesting movie generation for data_type='{data_type}'...")
    analyzer = NeuralDataAnalyzer()
    analyzer.load_sample_data()
    analyzer.preprocess_data()
    analyzer.create_epochs()
    if data_type == 'erp':
        # ERP needs only epochs
        pass
    elif data_type == 'pac':
        analyzer.time_frequency_analysis()
        analyzer.compute_phase_amplitude_coupling(low_freq=(4,8), high_freq=(30,40), method='tort')
    else:
        print(f"Unsupported data_type: {data_type}")
        return False
    movie_path = analyzer.generate_time_series_movie(data_type=data_type, output_path=output_path, fps=5, duration=2)
    if movie_path and os.path.exists(movie_path) and os.path.getsize(movie_path) > 0:
        print(f"PASS: Movie generated at {movie_path} ({os.path.getsize(movie_path)} bytes)")
        return True
    else:
        print(f"FAIL: Movie not generated or file is empty for {data_type}")
        return False

def main():
    all_passed = True
    all_passed &= test_movie_generation('erp', 'test_movie_erp.mp4')
    all_passed &= test_movie_generation('pac', 'test_movie_pac.mp4')
    if all_passed:
        print("\nAll movie generation tests PASSED.")
    else:
        print("\nSome movie generation tests FAILED.")

if __name__ == "__main__":
    main() 