"""
Advanced Data Management Module for Neural Data Analysis Pipeline
Provides BIDS format support, database integration, and data organization
"""

import os
import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import shutil
import hashlib
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BIDSMetadata:
    """BIDS metadata structure"""
    Name: str
    BIDSVersion: str = "1.7.0"
    DatasetType: str = "raw"
    Authors: List[str] = None
    HowToAcknowledge: str = ""
    Funding: List[str] = None
    EthicsApprovals: List[str] = None
    ReferencesAndLinks: List[str] = None
    DatasetDOI: str = ""

@dataclass
class BIDSParticipant:
    """BIDS participant information"""
    participant_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    handedness: Optional[str] = None
    group: Optional[str] = None
    comments: Optional[str] = None

class BIDSManager:
    """Manage BIDS format data organization"""
    
    def __init__(self, bids_root: str):
        self.bids_root = Path(bids_root)
        self.bids_root.mkdir(parents=True, exist_ok=True)
        self.setup_bids_structure()
        
    def setup_bids_structure(self):
        """Setup basic BIDS directory structure"""
        # Create required directories
        (self.bids_root / "participants.tsv").touch()
        (self.bids_root / "dataset_description.json").touch()
        
        # Create subdirectories
        (self.bids_root / "sub-01").mkdir(exist_ok=True)
        (self.bids_root / "sub-01" / "eeg").mkdir(exist_ok=True)
        (self.bids_root / "sub-01" / "anat").mkdir(exist_ok=True)
        (self.bids_root / "sub-01" / "func").mkdir(exist_ok=True)
        
        # Create derivatives directory
        (self.bids_root / "derivatives").mkdir(exist_ok=True)
        
        # Initialize dataset description
        self.create_dataset_description()
        
    def create_dataset_description(self, metadata: BIDSMetadata = None):
        """Create or update dataset description"""
        if metadata is None:
            metadata = BIDSMetadata(
                Name="Neural Data Analysis Dataset",
                Authors=["Data Pipeline"],
                HowToAcknowledge="Please cite this dataset if used in research"
            )
        
        description_file = self.bids_root / "dataset_description.json"
        with open(description_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def add_participant(self, participant: BIDSParticipant):
        """Add participant to BIDS dataset"""
        participants_file = self.bids_root / "participants.tsv"
        
        # Load existing participants
        if participants_file.exists() and participants_file.stat().st_size > 0:
            participants_df = pd.read_csv(participants_file, sep='\t')
        else:
            participants_df = pd.DataFrame(columns=['participant_id', 'age', 'sex', 'handedness', 'group', 'comments'])
        
        # Add new participant
        participant_data = {
            'participant_id': f"sub-{participant.participant_id}",
            'age': participant.age,
            'sex': participant.sex,
            'handedness': participant.handedness,
            'group': participant.group,
            'comments': participant.comments
        }
        
        # Check if participant already exists
        existing = participants_df[participants_df['participant_id'] == participant_data['participant_id']]
        if len(existing) > 0:
            # Update existing participant
            for key, value in participant_data.items():
                participants_df.loc[existing.index, key] = value
        else:
            # Add new participant
            participants_df = pd.concat([participants_df, pd.DataFrame([participant_data])], ignore_index=True)
        
        # Save updated participants file
        participants_df.to_csv(participants_file, sep='\t', index=False)
        
        # Create participant directory
        participant_dir = self.bids_root / f"sub-{participant.participant_id}"
        participant_dir.mkdir(exist_ok=True)
        (participant_dir / "eeg").mkdir(exist_ok=True)
        (participant_dir / "anat").mkdir(exist_ok=True)
        (participant_dir / "func").mkdir(exist_ok=True)
        
        logger.info(f"Participant {participant.participant_id} added to BIDS dataset")
    
    def add_eeg_data(self, participant_id: str, session: str, task: str, 
                    data_file: str, data_type: str = "eeg"):
        """Add EEG data to BIDS structure"""
        participant_dir = self.bids_root / f"sub-{participant_id}"
        session_dir = participant_dir / f"ses-{session}" if session else participant_dir
        session_dir.mkdir(exist_ok=True)
        
        eeg_dir = session_dir / "eeg"
        eeg_dir.mkdir(exist_ok=True)
        
        # Generate BIDS filename
        filename = f"sub-{participant_id}"
        if session:
            filename += f"_ses-{session}"
        filename += f"_task-{task}_{data_type}"
        
        # Copy and rename file
        source_file = Path(data_file)
        if source_file.exists():
            # Determine file extension
            if source_file.suffix.lower() in ['.edf', '.bdf', '.gdf']:
                extension = source_file.suffix.lower()
            else:
                extension = '.edf'  # Default
            
            bids_file = eeg_dir / f"{filename}{extension}"
            shutil.copy2(source_file, bids_file)
            
            # Create sidecar JSON file
            sidecar_file = eeg_dir / f"{filename}.json"
            self.create_eeg_sidecar(sidecar_file, data_type)
            
            logger.info(f"EEG data added: {bids_file}")
            return str(bids_file)
        
        return None
    
    def create_eeg_sidecar(self, sidecar_file: Path, data_type: str):
        """Create EEG sidecar JSON file"""
        sidecar_data = {
            "TaskName": "Unknown",
            "EEGReference": "Unknown",
            "SamplingFrequency": 1000,
            "PowerLineFrequency": 50,
            "SoftwareFilters": "Unknown",
            "CapManufacturer": "Unknown",
            "CapManufacturersModelName": "Unknown",
            "EEGChannelCount": 0,
            "EOGChannelCount": 0,
            "ECGChannelCount": 0,
            "EMGChannelCount": 0,
            "MiscChannelCount": 0,
            "TriggerChannelCount": 0,
            "RecordingDuration": 0,
            "RecordingType": "continuous",
            "EpochLength": 0,
            "DeviceSerialNumber": "Unknown",
            "InstitutionName": "Unknown",
            "InstitutionAddress": "Unknown",
            "InstitutionalDepartmentName": "Unknown"
        }
        
        with open(sidecar_file, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
    
    def add_derivative(self, participant_id: str, session: str, task: str,
                      derivative_type: str, data_file: str, description: str = ""):
        """Add derivative data to BIDS structure"""
        derivatives_dir = self.bids_root / "derivatives" / derivative_type
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        
        participant_dir = derivatives_dir / f"sub-{participant_id}"
        session_dir = participant_dir / f"ses-{session}" if session else participant_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate derivative filename
        filename = f"sub-{participant_id}"
        if session:
            filename += f"_ses-{session}"
        filename += f"_task-{task}_desc-{derivative_type}"
        
        # Copy file
        source_file = Path(data_file)
        if source_file.exists():
            extension = source_file.suffix
            derivative_file = session_dir / f"{filename}{extension}"
            shutil.copy2(source_file, derivative_file)
            
            # Create sidecar JSON
            sidecar_file = session_dir / f"{filename}.json"
            self.create_derivative_sidecar(sidecar_file, derivative_type, description)
            
            logger.info(f"Derivative added: {derivative_file}")
            return str(derivative_file)
        
        return None
    
    def create_derivative_sidecar(self, sidecar_file: Path, derivative_type: str, description: str):
        """Create derivative sidecar JSON file"""
        sidecar_data = {
            "Description": description,
            "DerivativeType": derivative_type,
            "GeneratedBy": {
                "Name": "Neural Data Pipeline",
                "Version": "1.0.0",
                "Description": "Generated by neural data analysis pipeline"
            },
            "GeneratedAt": datetime.now().isoformat()
        }
        
        with open(sidecar_file, 'w') as f:
            json.dump(sidecar_data, f, indent=2)
    
    def validate_bids_structure(self) -> Dict[str, Any]:
        """Validate BIDS dataset structure"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'files_found': [],
            'participants': []
        }
        
        # Check required files
        required_files = [
            "dataset_description.json",
            "participants.tsv"
        ]
        
        for file_name in required_files:
            file_path = self.bids_root / file_name
            if not file_path.exists():
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required file: {file_name}")
        
        # Check dataset description
        desc_file = self.bids_root / "dataset_description.json"
        if desc_file.exists():
            try:
                with open(desc_file, 'r') as f:
                    desc_data = json.load(f)
                
                if 'Name' not in desc_data:
                    validation_result['warnings'].append("Dataset description missing 'Name' field")
                
                if 'BIDSVersion' not in desc_data:
                    validation_result['warnings'].append("Dataset description missing 'BIDSVersion' field")
                    
            except json.JSONDecodeError:
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid JSON in dataset_description.json")
        
        # Check participants
        participants_file = self.bids_root / "participants.tsv"
        if participants_file.exists():
            try:
                participants_df = pd.read_csv(participants_file, sep='\t')
                validation_result['participants'] = participants_df['participant_id'].tolist()
                
                # Check participant directories
                for participant_id in participants_df['participant_id']:
                    participant_dir = self.bids_root / participant_id
                    if not participant_dir.exists():
                        validation_result['warnings'].append(f"Missing directory for {participant_id}")
                        
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Error reading participants.tsv: {e}")
        
        # Find all data files
        for file_path in self.bids_root.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.edf', '.bdf', '.gdf', '.json', '.tsv']:
                validation_result['files_found'].append(str(file_path.relative_to(self.bids_root)))
        
        return validation_result

class DatabaseManager:
    """Manage SQLite database for data tracking and metadata"""
    
    def __init__(self, db_path: str = "neural_data.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    file_hash TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    participant_id TEXT,
                    session_id TEXT,
                    task_id TEXT,
                    metadata TEXT,
                    status TEXT DEFAULT 'uploaded'
                )
            ''')
            
            # Create analysis_results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    analysis_type TEXT NOT NULL,
                    parameters TEXT,
                    results TEXT,
                    execution_time REAL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'completed',
                    FOREIGN KEY (file_id) REFERENCES files (id)
                )
            ''')
            
            # Create participants table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS participants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    participant_id TEXT UNIQUE NOT NULL,
                    age INTEGER,
                    sex TEXT,
                    handedness TEXT,
                    group_name TEXT,
                    comments TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    participant_id TEXT,
                    session_date DATE,
                    session_type TEXT,
                    notes TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (participant_id) REFERENCES participants (participant_id)
                )
            ''')
            
            # Create workflows table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_name TEXT NOT NULL,
                    workflow_config TEXT,
                    execution_log TEXT,
                    status TEXT DEFAULT 'created',
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_date TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def add_file(self, file_path: str, participant_id: str = None, 
                session_id: str = None, task_id: str = None, metadata: Dict = None):
        """Add file to database"""
        file_path = Path(file_path)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate file hash
            file_hash = self.calculate_file_hash(file_path)
            
            cursor.execute('''
                INSERT OR REPLACE INTO files 
                (file_path, file_name, file_type, file_size, file_hash, 
                 participant_id, session_id, task_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(file_path),
                file_path.name,
                file_path.suffix.lower(),
                file_path.stat().st_size,
                file_hash,
                participant_id,
                session_id,
                task_id,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def add_analysis_result(self, file_id: int, analysis_type: str, 
                          parameters: Dict, results: Dict, execution_time: float):
        """Add analysis result to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_results 
                (file_id, analysis_type, parameters, results, execution_time)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                file_id,
                analysis_type,
                json.dumps(parameters),
                json.dumps(results),
                execution_time
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_files_by_participant(self, participant_id: str) -> List[Dict]:
        """Get all files for a participant"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM files WHERE participant_id = ?
            ''', (participant_id,))
            
            columns = [description[0] for description in cursor.description]
            files = []
            
            for row in cursor.fetchall():
                file_dict = dict(zip(columns, row))
                if file_dict['metadata']:
                    file_dict['metadata'] = json.loads(file_dict['metadata'])
                files.append(file_dict)
            
            return files
    
    def get_analysis_results(self, file_id: int = None, analysis_type: str = None) -> List[Dict]:
        """Get analysis results"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT ar.*, f.file_name, f.participant_id 
                FROM analysis_results ar
                JOIN files f ON ar.file_id = f.id
                WHERE 1=1
            '''
            params = []
            
            if file_id:
                query += ' AND ar.file_id = ?'
                params.append(file_id)
            
            if analysis_type:
                query += ' AND ar.analysis_type = ?'
                params.append(analysis_type)
            
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                result_dict = dict(zip(columns, row))
                if result_dict['parameters']:
                    result_dict['parameters'] = json.loads(result_dict['parameters'])
                if result_dict['results']:
                    result_dict['results'] = json.loads(result_dict['results'])
                results.append(result_dict)
            
            return results
    
    def add_participant(self, participant_id: str, age: int = None, 
                       sex: str = None, handedness: str = None, 
                       group_name: str = None, comments: str = None):
        """Add participant to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO participants 
                (participant_id, age, sex, handedness, group_name, comments)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (participant_id, age, sex, handedness, group_name, comments))
            
            conn.commit()
    
    def get_database_summary(self) -> Dict[str, Any]:
        """Get database summary statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            summary = {}
            
            # Count files
            cursor.execute('SELECT COUNT(*) FROM files')
            summary['total_files'] = cursor.fetchone()[0]
            
            # Count participants
            cursor.execute('SELECT COUNT(*) FROM participants')
            summary['total_participants'] = cursor.fetchone()[0]
            
            # Count analysis results
            cursor.execute('SELECT COUNT(*) FROM analysis_results')
            summary['total_analyses'] = cursor.fetchone()[0]
            
            # Count workflows
            cursor.execute('SELECT COUNT(*) FROM workflows')
            summary['total_workflows'] = cursor.fetchone()[0]
            
            # File types distribution
            cursor.execute('''
                SELECT file_type, COUNT(*) 
                FROM files 
                GROUP BY file_type
            ''')
            summary['file_types'] = dict(cursor.fetchall())
            
            # Analysis types distribution
            cursor.execute('''
                SELECT analysis_type, COUNT(*) 
                FROM analysis_results 
                GROUP BY analysis_type
            ''')
            summary['analysis_types'] = dict(cursor.fetchall())
            
            return summary

class DataOrganizer:
    """Organize and manage data files with metadata tracking"""
    
    def __init__(self, base_dir: str = "organized_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def organize_by_participant(self, source_files: List[str], 
                              participant_mapping: Dict[str, str]):
        """Organize files by participant"""
        for file_path in source_files:
            file_path = Path(file_path)
            if file_path.exists():
                # Get participant ID from mapping
                participant_id = participant_mapping.get(file_path.name, "unknown")
                
                # Create participant directory
                participant_dir = self.base_dir / f"participant_{participant_id}"
                participant_dir.mkdir(exist_ok=True)
                
                # Copy file to participant directory
                dest_file = participant_dir / file_path.name
                shutil.copy2(file_path, dest_file)
                
                logger.info(f"Organized {file_path.name} -> {dest_file}")
    
    def organize_by_date(self, source_files: List[str]):
        """Organize files by date"""
        for file_path in source_files:
            file_path = Path(file_path)
            if file_path.exists():
                # Get file modification date
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                date_str = mod_time.strftime("%Y-%m-%d")
                
                # Create date directory
                date_dir = self.base_dir / date_str
                date_dir.mkdir(exist_ok=True)
                
                # Copy file to date directory
                dest_file = date_dir / file_path.name
                shutil.copy2(file_path, dest_file)
                
                logger.info(f"Organized {file_path.name} -> {dest_file}")
    
    def organize_by_type(self, source_files: List[str]):
        """Organize files by type"""
        for file_path in source_files:
            file_path = Path(file_path)
            if file_path.exists():
                # Get file extension
                file_type = file_path.suffix.lower().lstrip('.')
                if not file_type:
                    file_type = "unknown"
                
                # Create type directory
                type_dir = self.base_dir / file_type
                type_dir.mkdir(exist_ok=True)
                
                # Copy file to type directory
                dest_file = type_dir / file_path.name
                shutil.copy2(file_path, dest_file)
                
                logger.info(f"Organized {file_path.name} -> {dest_file}")
    
    def create_backup(self, backup_name: str = None):
        """Create backup of organized data"""
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = self.base_dir.parent / backup_name
        shutil.copytree(self.base_dir, backup_dir)
        
        logger.info(f"Backup created: {backup_dir}")
        return str(backup_dir)
    
    def cleanup_duplicates(self):
        """Remove duplicate files based on content hash"""
        file_hashes = {}
        duplicates = []
        
        # Calculate hashes for all files
        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file():
                file_hash = self.calculate_file_hash(file_path)
                
                if file_hash in file_hashes:
                    duplicates.append(file_path)
                else:
                    file_hashes[file_hash] = file_path
        
        # Remove duplicates
        for duplicate in duplicates:
            duplicate.unlink()
            logger.info(f"Removed duplicate: {duplicate}")
        
        logger.info(f"Removed {len(duplicates)} duplicate files")
        return len(duplicates)
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def get_organization_summary(self) -> Dict[str, Any]:
        """Get summary of organized data"""
        summary = {
            'total_files': 0,
            'total_size': 0,
            'file_types': {},
            'participants': set(),
            'dates': set()
        }
        
        for file_path in self.base_dir.rglob("*"):
            if file_path.is_file():
                summary['total_files'] += 1
                summary['total_size'] += file_path.stat().st_size
                
                # File type
                file_type = file_path.suffix.lower()
                summary['file_types'][file_type] = summary['file_types'].get(file_type, 0) + 1
                
                # Participant (if in path)
                if 'participant_' in str(file_path):
                    participant = str(file_path).split('participant_')[1].split('/')[0]
                    summary['participants'].add(participant)
                
                # Date (if in path)
                if '/' in str(file_path.relative_to(self.base_dir)):
                    date_part = str(file_path.relative_to(self.base_dir)).split('/')[0]
                    if len(date_part) == 10 and date_part.count('-') == 2:  # YYYY-MM-DD format
                        summary['dates'].add(date_part)
        
        # Convert sets to lists for JSON serialization
        summary['participants'] = list(summary['participants'])
        summary['dates'] = list(summary['dates'])
        
        return summary 