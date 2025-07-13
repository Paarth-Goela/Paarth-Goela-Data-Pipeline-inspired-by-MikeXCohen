"""
Workflow Automation Module for Neural Data Analysis
Provides reproducible pipeline execution and configuration management
"""

import os
import json
import yaml
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import subprocess
import sys
from dataclasses import dataclass, asdict
import pickle
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    name: str
    function: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    timeout: int = 3600  # 1 hour default
    retry_count: int = 3
    enabled: bool = True

@dataclass
class WorkflowConfig:
    """Configuration for a complete workflow"""
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    input_paths: List[str]
    output_path: str
    environment: Dict[str, str] = None
    metadata: Dict[str, Any] = None

class WorkflowExecutor:
    """Execute reproducible workflows with tracking and error handling"""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.execution_log = []
        self.step_results = {}
        self.setup_workspace()
        
    def setup_workspace(self):
        """Setup workspace for workflow execution"""
        # Create output directory
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        
        # Create execution log directory
        log_dir = Path(self.config.output_path) / "execution_logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create cache directory
        cache_dir = Path(self.config.output_path) / "cache"
        cache_dir.mkdir(exist_ok=True)
        
    def calculate_step_hash(self, step: WorkflowStep) -> str:
        """Calculate hash for step to determine if it needs re-execution"""
        step_data = {
            'name': step.name,
            'function': step.function,
            'parameters': step.parameters,
            'dependencies': step.dependencies or []
        }
        
        # Add dependency results to hash
        for dep in step.dependencies or []:
            if dep in self.step_results:
                step_data[f'dep_{dep}'] = self.step_results[dep].get('hash', '')
        
        step_str = json.dumps(step_data, sort_keys=True)
        return hashlib.sha256(step_str.encode()).hexdigest()
    
    def is_step_cached(self, step: WorkflowStep, step_hash: str) -> bool:
        """Check if step result is cached"""
        cache_file = Path(self.config.output_path) / "cache" / f"{step.name}_{step_hash}.pkl"
        return cache_file.exists()
    
    def load_cached_result(self, step: WorkflowStep, step_hash: str) -> Dict[str, Any]:
        """Load cached result for step"""
        cache_file = Path(self.config.output_path) / "cache" / f"{step.name}_{step_hash}.pkl"
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def save_cached_result(self, step: WorkflowStep, step_hash: str, result: Dict[str, Any]):
        """Save step result to cache"""
        cache_file = Path(self.config.output_path) / "cache" / f"{step.name}_{step_hash}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
    
    def execute_step(self, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step"""
        start_time = datetime.now()
        
        # Calculate step hash
        step_hash = self.calculate_step_hash(step)
        
        # Check cache first
        if self.is_step_cached(step, step_hash):
            logger.info(f"Using cached result for step: {step.name}")
            cached_result = self.load_cached_result(step, step_hash)
            cached_result['cached'] = True
            return cached_result
        
        # Execute step
        result = {
            'step_name': step.name,
            'start_time': start_time.isoformat(),
            'hash': step_hash,
            'cached': False,
            'success': False,
            'output': None,
            'error': None,
            'execution_time': 0
        }
        
        try:
            # Import and execute function
            module_name, function_name = step.function.rsplit('.', 1)
            module = __import__(module_name, fromlist=[function_name])
            func = getattr(module, function_name)
            
            # Execute with parameters
            output = func(**step.parameters)
            
            result['success'] = True
            result['output'] = output
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error executing step {step.name}: {e}")
        
        result['execution_time'] = (datetime.now() - start_time).total_seconds()
        
        # Cache successful results
        if result['success']:
            self.save_cached_result(step, step_hash, result)
        
        return result
    
    def execute_workflow(self) -> Dict[str, Any]:
        """Execute the complete workflow"""
        logger.info(f"Starting workflow: {self.config.name}")
        
        workflow_result = {
            'workflow_name': self.config.name,
            'start_time': datetime.now().isoformat(),
            'steps_executed': 0,
            'steps_successful': 0,
            'steps_failed': 0,
            'total_execution_time': 0,
            'step_results': {},
            'errors': []
        }
        
        start_time = datetime.now()
        
        # Execute steps in dependency order
        executed_steps = set()
        
        while len(executed_steps) < len(self.config.steps):
            progress_made = False
            
            for step in self.config.steps:
                if step.name in executed_steps or not step.enabled:
                    continue
                
                # Check dependencies
                if step.dependencies:
                    dependencies_met = all(dep in executed_steps for dep in step.dependencies)
                    if not dependencies_met:
                        continue
                
                # Execute step
                logger.info(f"Executing step: {step.name}")
                step_result = self.execute_step(step)
                self.step_results[step.name] = step_result
                workflow_result['step_results'][step.name] = step_result
                
                executed_steps.add(step.name)
                workflow_result['steps_executed'] += 1
                
                if step_result['success']:
                    workflow_result['steps_successful'] += 1
                else:
                    workflow_result['steps_failed'] += 1
                    workflow_result['errors'].append(f"Step {step.name}: {step_result['error']}")
                
                progress_made = True
            
            if not progress_made:
                # Circular dependency or missing step
                remaining_steps = [s.name for s in self.config.steps if s.name not in executed_steps and s.enabled]
                workflow_result['errors'].append(f"Circular dependency or missing step detected. Remaining: {remaining_steps}")
                break
        
        workflow_result['total_execution_time'] = (datetime.now() - start_time).total_seconds()
        workflow_result['end_time'] = datetime.now().isoformat()
        
        # Save execution log
        self.save_execution_log(workflow_result)
        
        logger.info(f"Workflow completed: {workflow_result['steps_successful']}/{workflow_result['steps_executed']} steps successful")
        
        return workflow_result
    
    def save_execution_log(self, workflow_result: Dict[str, Any]):
        """Save execution log to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = Path(self.config.output_path) / "execution_logs" / f"workflow_log_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(workflow_result, f, indent=2, default=str)
        
        logger.info(f"Execution log saved: {log_file}")

class WorkflowManager:
    """Manage multiple workflows and configurations"""
    
    def __init__(self, workflows_dir: str = "workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(exist_ok=True)
        
    def save_workflow_config(self, config: WorkflowConfig):
        """Save workflow configuration to file"""
        config_file = self.workflows_dir / f"{config.name}.yaml"
        
        # Convert dataclass to dict
        config_dict = asdict(config)
        
        # Convert steps to serializable format
        config_dict['steps'] = [
            {
                'name': step.name,
                'function': step.function,
                'parameters': step.parameters,
                'dependencies': step.dependencies,
                'timeout': step.timeout,
                'retry_count': step.retry_count,
                'enabled': step.enabled
            }
            for step in config.steps
        ]
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        logger.info(f"Workflow config saved: {config_file}")
    
    def load_workflow_config(self, workflow_name: str) -> WorkflowConfig:
        """Load workflow configuration from file"""
        config_file = self.workflows_dir / f"{workflow_name}.yaml"
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert steps back to WorkflowStep objects
        steps = []
        for step_dict in config_dict['steps']:
            step = WorkflowStep(
                name=step_dict['name'],
                function=step_dict['function'],
                parameters=step_dict['parameters'],
                dependencies=step_dict.get('dependencies'),
                timeout=step_dict.get('timeout', 3600),
                retry_count=step_dict.get('retry_count', 3),
                enabled=step_dict.get('enabled', True)
            )
            steps.append(step)
        
        config = WorkflowConfig(
            name=config_dict['name'],
            description=config_dict['description'],
            version=config_dict['version'],
            steps=steps,
            input_paths=config_dict['input_paths'],
            output_path=config_dict['output_path'],
            environment=config_dict.get('environment'),
            metadata=config_dict.get('metadata')
        )
        
        return config
    
    def list_workflows(self) -> List[str]:
        """List available workflows"""
        workflow_files = list(self.workflows_dir.glob("*.yaml"))
        return [f.stem for f in workflow_files]
    
    def create_standard_workflow(self, name: str, input_paths: List[str], 
                                output_path: str) -> WorkflowConfig:
        """Create a standard neural data analysis workflow"""
        
        steps = [
            WorkflowStep(
                name="data_validation",
                function="data_pipeline.DataValidator.validate_dataset_consistency",
                parameters={"file_paths": input_paths},
                dependencies=[]
            ),
            WorkflowStep(
                name="preprocessing",
                function="neural_backend.NeuralDataAnalyzer.preprocess_data",
                parameters={"file_paths": input_paths},
                dependencies=["data_validation"]
            ),
            WorkflowStep(
                name="spectral_analysis",
                function="neural_backend.NeuralDataAnalyzer.compute_spectral_features",
                parameters={},
                dependencies=["preprocessing"]
            ),
            WorkflowStep(
                name="connectivity_analysis",
                function="advanced_analysis.AdvancedAnalyzer.compute_connectivity",
                parameters={},
                dependencies=["preprocessing"]
            ),
            WorkflowStep(
                name="generate_report",
                function="report_generator.generate_analysis_report",
                parameters={"output_path": output_path},
                dependencies=["spectral_analysis", "connectivity_analysis"]
            )
        ]
        
        config = WorkflowConfig(
            name=name,
            description="Standard neural data analysis workflow",
            version="1.0.0",
            steps=steps,
            input_paths=input_paths,
            output_path=output_path,
            metadata={
                "created_by": "WorkflowManager",
                "created_date": datetime.now().isoformat(),
                "workflow_type": "standard_neural_analysis"
            }
        )
        
        return config

class ReproducibilityManager:
    """Manage reproducibility aspects of workflows"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def capture_environment(self) -> Dict[str, Any]:
        """Capture current environment for reproducibility"""
        environment = {
            'python_version': sys.version,
            'platform': sys.platform,
            'packages': self._get_installed_packages(),
            'environment_variables': dict(os.environ),
            'capture_time': datetime.now().isoformat()
        }
        
        return environment
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get list of installed packages and versions"""
        try:
            import pkg_resources
            return {dist.project_name: dist.version 
                   for dist in pkg_resources.working_set}
        except:
            return {}
    
    def save_environment_snapshot(self, workflow_name: str):
        """Save environment snapshot for workflow"""
        environment = self.capture_environment()
        
        env_file = self.output_path / f"{workflow_name}_environment.json"
        with open(env_file, 'w') as f:
            json.dump(environment, f, indent=2)
        
        logger.info(f"Environment snapshot saved: {env_file}")
    
    def create_reproducibility_report(self, workflow_result: Dict[str, Any], 
                                    workflow_config: WorkflowConfig) -> str:
        """Create reproducibility report"""
        report = {
            'workflow_info': {
                'name': workflow_config.name,
                'version': workflow_config.version,
                'description': workflow_config.description
            },
            'execution_info': {
                'start_time': workflow_result['start_time'],
                'end_time': workflow_result['end_time'],
                'total_execution_time': workflow_result['total_execution_time'],
                'steps_executed': workflow_result['steps_executed'],
                'steps_successful': workflow_result['steps_successful'],
                'steps_failed': workflow_result['steps_failed']
            },
            'environment': self.capture_environment(),
            'input_files': workflow_config.input_paths,
            'output_files': list(self.output_path.rglob("*")),
            'step_details': workflow_result['step_results']
        }
        
        report_file = self.output_path / f"{workflow_config.name}_reproducibility_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return str(report_file)

# Example workflow functions
def example_preprocessing_function(file_paths: List[str]) -> Dict[str, Any]:
    """Example preprocessing function for workflow"""
    # This would be implemented based on your specific needs
    return {
        'processed_files': file_paths,
        'preprocessing_parameters': {
            'filter_low': 1,
            'filter_high': 40,
            'notch_freq': 50
        }
    }

def example_analysis_function() -> Dict[str, Any]:
    """Example analysis function for workflow"""
    return {
        'analysis_results': 'example_results',
        'parameters': {}
    }

def example_report_function(output_path: str) -> str:
    """Example report generation function for workflow"""
    report_content = f"Analysis completed at {datetime.now()}"
    report_file = Path(output_path) / "analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    return str(report_file) 