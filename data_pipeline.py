"""
Advanced Data Pipeline Module for Neural Data Analysis
Focuses on batch processing, automation, and data management
"""

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
from dataclasses import dataclass, asdict
import hashlib
import zipfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for data pipeline processing"""
    input_dir: str
    output_dir: str
    file_patterns: List[str]
    preprocessing_steps: List[str]
    analysis_types: List[str]
    parallel_processing: bool = True
    max_workers: int = 4
    save_intermediate: bool = True
    compression: bool = True
    metadata_tracking: bool = True

@dataclass
class ProcessingResult:
    """Result of data processing"""
    file_path: str
    success: bool
    processing_time: float
    output_files: List[str]
    metadata: Dict[str, Any]
    errors: List[str] = None

class DataPipeline:
    """Advanced data pipeline for batch processing neural data"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results = []
        self.metadata_db = {}
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.output_dir}/intermediate").mkdir(exist_ok=True)
        Path(f"{self.config.output_dir}/logs").mkdir(exist_ok=True)
        Path(f"{self.config.output_dir}/metadata").mkdir(exist_ok=True)
        
    def find_data_files(self) -> List[str]:
        """Find all data files matching patterns"""
        files = []
        for pattern in self.config.file_patterns:
            for file_path in Path(self.config.input_dir).rglob(pattern):
                files.append(str(file_path))
        return sorted(files)
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for tracking"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def load_metadata(self) -> Dict[str, Any]:
        """Load existing metadata database"""
        metadata_file = f"{self.config.output_dir}/metadata/pipeline_metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self):
        """Save metadata database"""
        metadata_file = f"{self.config.output_dir}/metadata/pipeline_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata_db, f, indent=2)
    
    def process_single_file(self, file_path: str) -> ProcessingResult:
        """Process a single data file"""
        start_time = datetime.now()
        errors = []
        output_files = []
        
        try:
            # Calculate file hash for tracking
            file_hash = self.calculate_file_hash(file_path)
            
            # Check if already processed
            if file_hash in self.metadata_db:
                logger.info(f"File already processed: {file_path}")
                return ProcessingResult(
                    file_path=file_path,
                    success=True,
                    processing_time=0.0,
                    output_files=self.metadata_db[file_hash].get('output_files', []),
                    metadata=self.metadata_db[file_hash]
                )
            
            # Extract file info
            file_info = {
                'filename': os.path.basename(file_path),
                'file_size': os.path.getsize(file_path),
                'file_hash': file_hash,
                'processing_date': datetime.now().isoformat(),
                'preprocessing_steps': self.config.preprocessing_steps,
                'analysis_types': self.config.analysis_types
            }
            
            # Process based on file type
            if file_path.endswith('.edf'):
                output_files = self._process_edf_file(file_path, file_info)
            elif file_path.endswith('.csv'):
                output_files = self._process_csv_file(file_path, file_info)
            elif file_path.endswith('.mat'):
                output_files = self._process_mat_file(file_path, file_info)
            else:
                errors.append(f"Unsupported file type: {file_path}")
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store metadata
            file_info['output_files'] = output_files
            file_info['processing_time'] = processing_time
            self.metadata_db[file_hash] = file_info
            
            return ProcessingResult(
                file_path=file_path,
                success=len(errors) == 0,
                processing_time=processing_time,
                output_files=output_files,
                metadata=file_info,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(
                file_path=file_path,
                success=False,
                processing_time=processing_time,
                output_files=[],
                metadata={},
                errors=errors
            )
    
    def _process_edf_file(self, file_path: str, file_info: Dict[str, Any]) -> List[str]:
        """Process EDF file"""
        output_files = []
        base_name = Path(file_path).stem
        
        try:
            # Import MNE here to avoid circular imports
            import mne
            
            # Load raw data
            raw = mne.io.read_raw_edf(file_path, preload=True)
            
            # Apply preprocessing steps
            if 'filter' in self.config.preprocessing_steps:
                raw.filter(1, 40)
            
            if 'notch' in self.config.preprocessing_steps:
                raw.notch_filter(50)
            
            # Save processed data
            processed_file = f"{self.config.output_dir}/intermediate/{base_name}_processed.fif"
            raw.save(processed_file, overwrite=True)
            output_files.append(processed_file)
            
            # Generate analysis files
            if 'spectral' in self.config.analysis_types:
                spectral_file = self._generate_spectral_analysis(raw, base_name)
                output_files.append(spectral_file)
            
            if 'erp' in self.config.analysis_types:
                erp_file = self._generate_erp_analysis(raw, base_name)
                output_files.append(erp_file)
                
        except Exception as e:
            logger.error(f"Error processing EDF file {file_path}: {e}")
            raise
            
        return output_files
    
    def _process_csv_file(self, file_path: str, file_info: Dict[str, Any]) -> List[str]:
        """Process CSV file"""
        output_files = []
        base_name = Path(file_path).stem
        
        try:
            # Load data
            data = pd.read_csv(file_path)
            
            # Basic preprocessing
            if 'clean' in self.config.preprocessing_steps:
                data = data.dropna()
            
            # Save processed data
            processed_file = f"{self.config.output_dir}/intermediate/{base_name}_processed.csv"
            data.to_csv(processed_file, index=False)
            output_files.append(processed_file)
            
            # Generate analysis files
            if 'statistics' in self.config.analysis_types:
                stats_file = self._generate_statistics(data, base_name)
                output_files.append(stats_file)
                
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}")
            raise
            
        return output_files
    
    def _process_mat_file(self, file_path: str, file_info: Dict[str, Any]) -> List[str]:
        """Process MAT file"""
        output_files = []
        base_name = Path(file_path).stem
        
        try:
            # Import scipy here to avoid circular imports
            from scipy import io
            
            # Load data
            mat_data = io.loadmat(file_path)
            
            # Convert to more accessible format
            data_dict = {}
            for key in mat_data.keys():
                if not key.startswith('__'):
                    data_dict[key] = mat_data[key]
            
            # Save as pickle for easier access
            processed_file = f"{self.config.output_dir}/intermediate/{base_name}_processed.pkl"
            with open(processed_file, 'wb') as f:
                pickle.dump(data_dict, f)
            output_files.append(processed_file)
            
        except Exception as e:
            logger.error(f"Error processing MAT file {file_path}: {e}")
            raise
            
        return output_files
    
    def _generate_spectral_analysis(self, raw, base_name: str) -> str:
        """Generate spectral analysis"""
        try:
            import mne
            
            # Compute PSD
            psds, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=40)
            
            # Save results
            spectral_file = f"{self.config.output_dir}/intermediate/{base_name}_spectral.npz"
            np.savez(spectral_file, psds=psds, freqs=freqs)
            
            return spectral_file
        except Exception as e:
            logger.error(f"Error generating spectral analysis: {e}")
            return ""
    
    def _generate_erp_analysis(self, raw, base_name: str) -> str:
        """Generate ERP analysis"""
        try:
            # This would require events/epochs - simplified for now
            erp_file = f"{self.config.output_dir}/intermediate/{base_name}_erp.npz"
            # Placeholder for ERP analysis
            np.savez(erp_file, data=np.array([]))
            return erp_file
        except Exception as e:
            logger.error(f"Error generating ERP analysis: {e}")
            return ""
    
    def _generate_statistics(self, data: pd.DataFrame, base_name: str) -> str:
        """Generate statistical summary"""
        try:
            stats = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'summary': data.describe().to_dict(),
                'missing_values': data.isnull().sum().to_dict()
            }
            
            stats_file = f"{self.config.output_dir}/intermediate/{base_name}_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            return stats_file
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return ""
    
    def run_pipeline(self) -> List[ProcessingResult]:
        """Run the complete data pipeline"""
        logger.info("Starting data pipeline...")
        
        # Load existing metadata
        self.metadata_db = self.load_metadata()
        
        # Find files to process
        files = self.find_data_files()
        logger.info(f"Found {len(files)} files to process")
        
        if self.config.parallel_processing:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_file = {executor.submit(self.process_single_file, file_path): file_path 
                                for file_path in files}
                
                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()
                    self.results.append(result)
                    
                    if result.success:
                        logger.info(f"Successfully processed: {result.file_path}")
                    else:
                        logger.error(f"Failed to process: {result.file_path} - {result.errors}")
        else:
            # Sequential processing
            for file_path in files:
                result = self.process_single_file(file_path)
                self.results.append(result)
                
                if result.success:
                    logger.info(f"Successfully processed: {result.file_path}")
                else:
                    logger.error(f"Failed to process: {result.file_path} - {result.errors}")
        
        # Save metadata
        self.save_metadata()
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info("Data pipeline completed!")
        return self.results
    
    def _generate_summary_report(self):
        """Generate summary report of pipeline execution"""
        total_files = len(self.results)
        successful_files = sum(1 for r in self.results if r.success)
        failed_files = total_files - successful_files
        total_time = sum(r.processing_time for r in self.results)
        
        summary = {
            'pipeline_config': asdict(self.config),
            'execution_summary': {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'success_rate': successful_files / total_files if total_files > 0 else 0,
                'total_processing_time': total_time,
                'average_processing_time': total_time / total_files if total_files > 0 else 0
            },
            'file_results': [
                {
                    'file_path': r.file_path,
                    'success': r.success,
                    'processing_time': r.processing_time,
                    'output_files': r.output_files,
                    'errors': r.errors
                }
                for r in self.results
            ]
        }
        
        # Save summary report
        summary_file = f"{self.config.output_dir}/pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary report saved: {summary_file}")
    
    def create_archive(self, archive_name: str = None) -> str:
        """Create compressed archive of processed data"""
        if archive_name is None:
            archive_name = f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        archive_path = f"{self.config.output_dir}/{archive_name}"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all output files
            for result in self.results:
                if result.success:
                    for output_file in result.output_files:
                        if os.path.exists(output_file):
                            zipf.write(output_file, os.path.relpath(output_file, self.config.output_dir))
            
            # Add metadata
            metadata_file = f"{self.config.output_dir}/metadata/pipeline_metadata.json"
            if os.path.exists(metadata_file):
                zipf.write(metadata_file, "metadata/pipeline_metadata.json")
        
        logger.info(f"Archive created: {archive_path}")
        return archive_path
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files to save space"""
        intermediate_dir = f"{self.config.output_dir}/intermediate"
        if os.path.exists(intermediate_dir):
            shutil.rmtree(intermediate_dir)
            os.makedirs(intermediate_dir)
            logger.info("Intermediate files cleaned up")

class BatchProcessor:
    """Batch processing utilities for data pipeline"""
    
    @staticmethod
    def create_batch_configs(input_dir: str, output_base: str, 
                           batch_size: int = 10) -> List[PipelineConfig]:
        """Create multiple batch configurations for large datasets"""
        configs = []
        
        # Find all data files
        all_files = []
        for pattern in ['*.edf', '*.csv', '*.mat']:
            all_files.extend(Path(input_dir).rglob(pattern))
        
        # Split into batches
        for i in range(0, len(all_files), batch_size):
            batch_files = all_files[i:i + batch_size]
            batch_output = f"{output_base}/batch_{i//batch_size + 1}"
            
            config = PipelineConfig(
                input_dir=input_dir,
                output_dir=batch_output,
                file_patterns=['*.edf', '*.csv', '*.mat'],
                preprocessing_steps=['filter', 'notch', 'clean'],
                analysis_types=['spectral', 'erp', 'statistics'],
                parallel_processing=True,
                max_workers=4
            )
            configs.append(config)
        
        return configs
    
    @staticmethod
    def run_batch_processing(configs: List[PipelineConfig]) -> List[List[ProcessingResult]]:
        """Run multiple batch configurations"""
        all_results = []
        
        for i, config in enumerate(configs):
            logger.info(f"Running batch {i+1}/{len(configs)}")
            pipeline = DataPipeline(config)
            results = pipeline.run_pipeline()
            all_results.append(results)
        
        return all_results

class DataValidator:
    """Data validation utilities for pipeline"""
    
    @staticmethod
    def validate_file_integrity(file_path: str) -> Dict[str, Any]:
        """Validate file integrity and basic properties"""
        validation_result = {
            'file_path': file_path,
            'exists': False,
            'readable': False,
            'file_size': 0,
            'file_hash': None,
            'errors': []
        }
        
        try:
            if not os.path.exists(file_path):
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            validation_result['exists'] = True
            validation_result['file_size'] = os.path.getsize(file_path)
            
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Read first 1KB
                validation_result['readable'] = True
            except Exception as e:
                validation_result['errors'].append(f"File not readable: {e}")
                return validation_result
            
            # Calculate hash
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            validation_result['file_hash'] = hash_sha256.hexdigest()
            
        except Exception as e:
            validation_result['errors'].append(f"Validation error: {e}")
        
        return validation_result
    
    @staticmethod
    def validate_dataset_consistency(file_paths: List[str]) -> Dict[str, Any]:
        """Validate consistency across multiple files in a dataset"""
        validation_results = {
            'total_files': len(file_paths),
            'valid_files': 0,
            'invalid_files': 0,
            'file_hashes': {},
            'duplicates': [],
            'errors': []
        }
        
        # Validate each file
        for file_path in file_paths:
            result = DataValidator.validate_file_integrity(file_path)
            if result['exists'] and result['readable']:
                validation_results['valid_files'] += 1
                file_hash = result['file_hash']
                
                if file_hash in validation_results['file_hashes']:
                    validation_results['duplicates'].append({
                        'file1': validation_results['file_hashes'][file_hash],
                        'file2': file_path,
                        'hash': file_hash
                    })
                else:
                    validation_results['file_hashes'][file_hash] = file_path
            else:
                validation_results['invalid_files'] += 1
                validation_results['errors'].extend(result['errors'])
        
        return validation_results 