#!/usr/bin/env python3
"""
Enhanced Data Upload System for NeurInSpectre
Supports industry-standard cybersecurity and AI/ML data formats
"""

import json
import yaml
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import pickle
import base64
import io
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import re
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Optional dependencies (keep this module importable in minimal installs)
try:
    import h5py  # type: ignore
    _H5PY_AVAILABLE = True
except Exception:
    h5py = None  # type: ignore[assignment]
    _H5PY_AVAILABLE = False

class IndustryStandardDataParser:
    """
    Comprehensive parser for cybersecurity and AI/ML data formats
    Supports: STIX 2.1 JSON, CSV, XML (STIX 1.x), YAML, NPY/NPZ, HDF5, PKL
    """
    
    def __init__(self):
        self.supported_formats = {
            'json': ['json'],
            'csv': ['csv'],
            'xml': ['xml'],
            'yaml': ['yaml', 'yml'],
            'numpy': ['npy', 'npz'],
            'hdf5': ['h5', 'hdf5'],
            'pickle': ['pkl', 'pickle'],
            'stix': ['stix', 'json'],  # STIX can be JSON or XML
            'mitre': ['json', 'xml'],  # MITRE ATT&CK data
            'adversarial': ['npy', 'npz', 'h5']  # Adversarial ML data
        }
        
        # STIX 2.1 object types for validation
        self.stix_object_types = {
            'attack-pattern', 'campaign', 'course-of-action', 'grouping',
            'identity', 'incident', 'indicator', 'infrastructure', 'intrusion-set',
            'location', 'malware', 'malware-analysis', 'note', 'observed-data',
            'opinion', 'report', 'threat-actor', 'tool', 'vulnerability'
        }
        
        # MITRE ATT&CK technique pattern
        self.mitre_technique_pattern = re.compile(r'T\d{4}(\.\d{3})?')
    
    def detect_data_type(self, data: Any, filename: str) -> Dict[str, Any]:
        """
        Intelligently detect the type of cybersecurity/AI data
        """
        detection_result = {
            'data_category': 'unknown',
            'format_type': 'unknown',
            'is_attack_data': False,
            'is_normal_data': False,
            'confidence': 0.0,
            'metadata': {}
        }
        
        try:
            # Check if it's STIX 2.1 format
            if isinstance(data, dict):
                if self._is_stix_21_data(data):
                    detection_result.update({
                        'data_category': 'threat_intelligence',
                        'format_type': 'stix_21',
                        'is_attack_data': True,
                        'confidence': 0.95,
                        'metadata': {
                            'stix_version': data.get('spec_version', '2.1'),
                            'object_count': len(data.get('objects', [])),
                            'object_types': list(set(obj.get('type') for obj in data.get('objects', [])))
                        }
                    })
                
                elif self._is_mitre_attack_data(data):
                    detection_result.update({
                        'data_category': 'mitre_attack',
                        'format_type': 'mitre_json',
                        'is_attack_data': True,
                        'confidence': 0.90,
                        'metadata': {
                            'techniques_count': len([k for k in data.keys() if self.mitre_technique_pattern.match(k)]),
                            'has_tactics': 'tactics' in str(data).lower(),
                            'has_techniques': 'techniques' in str(data).lower()
                        }
                    })
                
                elif self._is_adversarial_ml_data(data):
                    detection_result.update({
                        'data_category': 'adversarial_ml',
                        'format_type': 'adversarial_json',
                        'is_attack_data': True,
                        'confidence': 0.85,
                        'metadata': {
                            'has_gradients': 'gradient' in str(data).lower(),
                            'has_perturbations': 'perturbation' in str(data).lower(),
                            'attack_types': self._extract_attack_types(data)
                        }
                    })
            
            # Check if it's tabular threat intelligence
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                if self._is_threat_intel_csv(data):
                    detection_result.update({
                        'data_category': 'threat_intelligence',
                        'format_type': 'csv_threat_intel',
                        'is_attack_data': True,
                        'confidence': 0.80,
                        'metadata': {
                            'record_count': len(data),
                            'columns': list(data[0].keys()) if data else [],
                            'has_iocs': self._has_indicators_of_compromise(data)
                        }
                    })
            
            # Check if it's numpy adversarial data
            elif isinstance(data, np.ndarray):
                if self._is_adversarial_numpy(data, filename):
                    detection_result.update({
                        'data_category': 'adversarial_ml',
                        'format_type': 'numpy_adversarial',
                        'is_attack_data': True,
                        'confidence': 0.85,
                        'metadata': {
                            'shape': data.shape,
                            'dtype': str(data.dtype),
                            'size_mb': data.nbytes / 1024 / 1024,
                            'likely_type': self._infer_numpy_content_type(data, filename)
                        }
                    })
            
            # If no specific type detected, classify as normal data
            if detection_result['confidence'] < 0.5:
                detection_result.update({
                    'data_category': 'general',
                    'format_type': 'normal_data',
                    'is_normal_data': True,
                    'confidence': 0.60
                })
        
        except Exception as e:
            logger.warning(f"Error in data type detection: {e}")
            detection_result['metadata']['detection_error'] = str(e)
        
        return detection_result
    
    def _is_stix_21_data(self, data: dict) -> bool:
        """Check if data conforms to STIX 2.1 format"""
        try:
            # Check for STIX 2.1 bundle structure
            if data.get('type') == 'bundle' and 'objects' in data:
                spec_version = data.get('spec_version', '')
                if spec_version.startswith('2.'):
                    # Validate object types
                    objects = data.get('objects', [])
                    valid_objects = all(
                        obj.get('type') in self.stix_object_types 
                        for obj in objects if isinstance(obj, dict)
                    )
                    return valid_objects and len(objects) > 0
            
            # Check for individual STIX object
            elif data.get('type') in self.stix_object_types:
                return 'id' in data and 'created' in data
            
            return False
        except:
            return False
    
    def _is_mitre_attack_data(self, data: dict) -> bool:
        """Check if data contains MITRE ATT&CK framework information"""
        try:
            data_str = str(data).lower()
            
            # Check for MITRE ATT&CK indicators
            mitre_indicators = [
                'mitre', 'attack', 'technique', 'tactic', 'procedure',
                'sub-technique', 'matrix', 'enterprise', 'mobile', 'ics'
            ]
            
            indicator_count = sum(1 for indicator in mitre_indicators if indicator in data_str)
            
            # Check for technique IDs (T1234 or T1234.001)
            technique_matches = len(self.mitre_technique_pattern.findall(str(data)))
            
            return indicator_count >= 3 or technique_matches > 0
        except:
            return False
    
    def _is_adversarial_ml_data(self, data: dict) -> bool:
        """Check if data contains adversarial ML information"""
        try:
            data_str = str(data).lower()
            
            adversarial_keywords = [
                'adversarial', 'attack', 'perturbation', 'gradient', 'fgsm',
                'pgd', 'c&w', 'deepfool', 'evasion', 'poisoning', 'backdoor',
                'adversarial_examples', 'epsilon', 'loss_function'
            ]
            
            keyword_count = sum(1 for keyword in adversarial_keywords if keyword in data_str)
            return keyword_count >= 2
        except:
            return False
    
    def _is_threat_intel_csv(self, data: list) -> bool:
        """Check if CSV data contains threat intelligence"""
        try:
            if not data or not isinstance(data[0], dict):
                return False
            
            columns = set(str(k).lower() for k in data[0].keys())
            
            threat_intel_columns = {
                'ioc', 'indicator', 'hash', 'md5', 'sha1', 'sha256',
                'ip', 'domain', 'url', 'malware', 'threat', 'campaign',
                'apt', 'ttp', 'technique', 'tactic', 'severity', 'confidence'
            }
            
            matches = len(columns.intersection(threat_intel_columns))
            return matches >= 2
        except:
            return False
    
    def _has_indicators_of_compromise(self, data: list) -> bool:
        """Check if data contains Indicators of Compromise (IoCs)"""
        try:
            sample_data = str(data[:5]).lower()  # Check first 5 records
            
            ioc_patterns = [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IP addresses
                r'\b[a-f0-9]{32}\b',  # MD5 hashes
                r'\b[a-f0-9]{40}\b',  # SHA1 hashes
                r'\b[a-f0-9]{64}\b',  # SHA256 hashes
                r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\b'  # Domains
            ]
            
            return any(re.search(pattern, sample_data) for pattern in ioc_patterns)
        except:
            return False
    
    def _is_adversarial_numpy(self, data: np.ndarray, filename: str) -> bool:
        """Check if numpy data contains adversarial examples"""
        try:
            filename_lower = filename.lower()
            
            # Check filename for adversarial indicators
            adversarial_filenames = [
                'adversarial', 'attack', 'perturb', 'gradient', 'fgsm',
                'pgd', 'clean', 'poison', 'backdoor', 'evasion'
            ]
            
            filename_match = any(keyword in filename_lower for keyword in adversarial_filenames)
            
            # Check data characteristics
            # Adversarial examples often have specific shapes and value ranges
            is_image_like = len(data.shape) >= 3 and data.shape[-1] in [1, 3, 4]  # Channels
            is_gradient_like = len(data.shape) >= 2 and np.any(data < 0)  # Gradients can be negative
            is_normalized = np.all(data >= 0) and np.all(data <= 1)  # Normalized pixel values
            
            return filename_match or is_image_like or is_gradient_like
        except:
            return False
    
    def _infer_numpy_content_type(self, data: np.ndarray, filename: str) -> str:
        """Infer the specific type of numpy content"""
        try:
            filename_lower = filename.lower()
            
            if 'gradient' in filename_lower:
                return 'gradients'
            elif 'clean' in filename_lower:
                return 'clean_samples'
            elif 'adversarial' in filename_lower or 'attack' in filename_lower:
                return 'adversarial_samples'
            elif 'poison' in filename_lower:
                return 'poisoned_data'
            elif len(data.shape) == 4:  # Batch of images
                return 'image_batch'
            elif len(data.shape) == 3:  # Single image or feature maps
                return 'image_or_features'
            elif len(data.shape) == 2:  # Matrix
                return 'feature_matrix'
            else:
                return 'vector_data'
        except:
            return 'unknown_numpy'
    
    def _extract_attack_types(self, data: dict) -> List[str]:
        """Extract attack types from adversarial ML data"""
        try:
            data_str = str(data).lower()
            
            attack_types = []
            attack_keywords = {
                'fgsm': 'Fast Gradient Sign Method',
                'pgd': 'Projected Gradient Descent',
                'c&w': 'Carlini & Wagner',
                'deepfool': 'DeepFool',
                'jsma': 'Jacobian-based Saliency Map',
                'one_pixel': 'One Pixel Attack',
                'boundary': 'Boundary Attack',
                'hop_skip_jump': 'HopSkipJump',
                'evasion': 'Evasion Attack',
                'poisoning': 'Data Poisoning',
                'backdoor': 'Backdoor Attack',
                'membership_inference': 'Membership Inference',
                'model_extraction': 'Model Extraction'
            }
            
            for keyword, attack_name in attack_keywords.items():
                if keyword in data_str:
                    attack_types.append(attack_name)
            
            return attack_types
        except:
            return []

def parse_uploaded_file_enhanced(contents: str, filename: str) -> Tuple[Any, str, Dict[str, Any]]:
    """
    Enhanced file parser supporting all industry-standard formats
    Returns: (parsed_data, format_type, metadata)
    """
    parser = IndustryStandardDataParser()
    
    try:
        # Decode the uploaded content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        file_extension = filename.lower().split('.')[-1]
        
        # Parse based on file extension
        if file_extension == 'json':
            # JSON format - STIX 2.1, MITRE ATT&CK, or general JSON
            data = json.loads(decoded.decode('utf-8'))
            detection = parser.detect_data_type(data, filename)
            return data, f"json_{detection['format_type']}", detection
            
        elif file_extension == 'csv':
            # CSV format - tabular threat intelligence or general data
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            data = df.to_dict('records')
            detection = parser.detect_data_type(data, filename)
            return data, f"csv_{detection['format_type']}", detection
            
        elif file_extension == 'xml':
            # XML format - STIX 1.x or general XML
            root = ET.fromstring(decoded.decode('utf-8'))
            
            # Convert XML to dict for easier processing
            data = xml_to_dict(root)
            
            # Check if it's STIX 1.x
            if 'stix' in root.tag.lower() or any('stix' in str(child.tag).lower() for child in root):
                detection = {
                    'data_category': 'threat_intelligence',
                    'format_type': 'stix_1x',
                    'is_attack_data': True,
                    'confidence': 0.90,
                    'metadata': {
                        'root_tag': root.tag,
                        'namespace': root.tag.split('}')[0] if '}' in root.tag else None,
                        'child_count': len(root)
                    }
                }
            else:
                detection = parser.detect_data_type(data, filename)
            
            return data, f"xml_{detection['format_type']}", detection
            
        elif file_extension in ['yaml', 'yml']:
            # YAML format - configuration files, rules, or structured data
            data = yaml.safe_load(decoded.decode('utf-8'))
            detection = parser.detect_data_type(data, filename)
            
            # Check for common security rule formats
            if isinstance(data, dict):
                if 'rules' in data or 'signatures' in data:
                    detection.update({
                        'data_category': 'security_rules',
                        'format_type': 'yaml_rules',
                        'is_attack_data': True,
                        'confidence': 0.85
                    })
                elif 'config' in data or 'configuration' in data:
                    detection.update({
                        'data_category': 'configuration',
                        'format_type': 'yaml_config',
                        'is_normal_data': True,
                        'confidence': 0.80
                    })
            
            return data, f"yaml_{detection['format_type']}", detection
            
        elif file_extension == 'npy':
            # NumPy array format - adversarial examples, gradients, etc.
            data = np.load(io.BytesIO(decoded), allow_pickle=True)
            detection = parser.detect_data_type(data, filename)
            
            return data, f"numpy_{detection['format_type']}", detection
            
        elif file_extension == 'npz':
            # NumPy compressed format - multiple arrays
            npz_data = np.load(io.BytesIO(decoded), allow_pickle=True)
            data = {key: npz_data[key] for key in npz_data.files}
            
            # Analyze the content
            detection = {
                'data_category': 'adversarial_ml',
                'format_type': 'npz_multi_array',
                'is_attack_data': True,
                'confidence': 0.80,
                'metadata': {
                    'array_names': list(data.keys()),
                    'array_shapes': {key: arr.shape for key, arr in data.items()},
                    'total_size_mb': sum(arr.nbytes for arr in data.values()) / 1024 / 1024
                }
            }
            
            return data, f"npz_{detection['format_type']}", detection
            
        elif file_extension in ['h5', 'hdf5']:
            # HDF5 format - large-scale ML datasets
            if not _H5PY_AVAILABLE or h5py is None:
                return None, 'missing_dependency_hdf5', {
                    'data_category': 'error',
                    'format_type': 'missing_dependency',
                    'confidence': 0.0,
                    'metadata': {'error': 'h5py is required to parse .h5/.hdf5 files (pip install h5py)'}
                }
            with h5py.File(io.BytesIO(decoded), 'r') as f:
                data = {}
                metadata = {
                    'groups': [],
                    'datasets': [],
                    'total_size_mb': 0
                }
                
                def extract_hdf5_data(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name] = obj[...].tolist()  # Convert to list for JSON serialization
                        metadata['datasets'].append({
                            'name': name,
                            'shape': obj.shape,
                            'dtype': str(obj.dtype),
                            'size_mb': obj.nbytes / 1024 / 1024
                        })
                        metadata['total_size_mb'] += obj.nbytes / 1024 / 1024
                    elif isinstance(obj, h5py.Group):
                        metadata['groups'].append(name)
                
                f.visititems(extract_hdf5_data)
            
            detection = {
                'data_category': 'ml_dataset',
                'format_type': 'hdf5_dataset',
                'is_normal_data': True,
                'confidence': 0.85,
                'metadata': metadata
            }
            
            return data, f"hdf5_{detection['format_type']}", detection
            
        elif file_extension in ['pkl', 'pickle']:
            # Pickle format - Python objects
            data = pickle.loads(decoded)
            detection = parser.detect_data_type(data, filename)
            
            return data, f"pickle_{detection['format_type']}", detection

        elif file_extension == 'png':
            # PNG image: return lightweight metadata (no heavy image deps required)
            width = height = None
            try:
                # PNG header: signature (8) + IHDR chunk (length 4 + type 4 + data 13 + crc 4)
                if decoded[:8] == b"\x89PNG\r\n\x1a\n" and len(decoded) >= 24:
                    # IHDR chunk type should start at byte 12
                    if decoded[12:16] == b"IHDR" and len(decoded) >= 24:
                        width = int.from_bytes(decoded[16:20], "big")
                        height = int.from_bytes(decoded[20:24], "big")
            except Exception:
                width = height = None

            sha256 = None
            try:
                import hashlib
                sha256 = hashlib.sha256(decoded).hexdigest()
            except Exception:
                sha256 = None

            detection = {
                'data_category': 'image',
                'format_type': 'png_image',
                'is_attack_data': False,
                'is_normal_data': True,
                'confidence': 0.70,
                'metadata': {
                    'mime': 'image/png',
                    'size_bytes': int(len(decoded)),
                    'sha256': sha256,
                    'width': width,
                    'height': height,
                },
            }

            # Keep parsed_data small + JSON-serializable
            data = {
                'type': 'image/png',
                'width': width,
                'height': height,
                'sha256': sha256,
            }
            return data, "image_png", detection
            
        else:
            # Unsupported format
            return None, f'unsupported_format_{file_extension}', {
                'data_category': 'unknown',
                'format_type': 'unsupported',
                'confidence': 0.0,
                'metadata': {'error': f'Unsupported file extension: {file_extension}'}
            }
            
    except Exception as e:
        logger.error(f"Failed to parse uploaded file {filename}: {e}")
        return None, 'parsing_error', {
            'data_category': 'error',
            'format_type': 'parsing_failed',
            'confidence': 0.0,
            'metadata': {'error': str(e)}
        }

def xml_to_dict(element):
    """Convert XML element to dictionary"""
    result = {}
    
    # Add attributes
    if element.attrib:
        result['@attributes'] = element.attrib
    
    # Add text content
    if element.text and element.text.strip():
        if len(element) == 0:
            return element.text.strip()
        else:
            result['@text'] = element.text.strip()
    
    # Add child elements
    for child in element:
        child_data = xml_to_dict(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_data)
        else:
            result[child.tag] = child_data
    
    return result

def generate_upload_status_display(filename: str, format_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive status display for uploaded files
    """
    detection = metadata
    
    # Determine status color and icon
    if detection['confidence'] > 0.8:
        status_color = 'lightgreen'
        status_icon = '✅'
    elif detection['confidence'] > 0.6:
        status_color = 'yellow'
        status_icon = '⚠️'
    else:
        status_color = 'red'
        status_icon = '❌'
    
    # Create detailed status information
    status_info = {
        'filename': filename,
        'format_type': format_type,
        'status_color': status_color,
        'status_icon': status_icon,
        'data_category': detection.get('data_category', 'unknown'),
        'is_attack_data': detection.get('is_attack_data', False),
        'is_normal_data': detection.get('is_normal_data', False),
        'confidence': detection.get('confidence', 0.0),
        'metadata': detection.get('metadata', {})
    }
    
    return status_info

# Example usage and testing
if __name__ == "__main__":
    # Test the parser with sample data
    parser = IndustryStandardDataParser()
    
    # Test STIX 2.1 data
    stix_sample = {
        "type": "bundle",
        "id": "bundle--01234567-89ab-cdef-0123-456789abcdef",
        "spec_version": "2.1",
        "objects": [
            {
                "type": "indicator",
                "id": "indicator--01234567-89ab-cdef-0123-456789abcdef",
                "created": "2024-01-01T00:00:00.000Z",
                "modified": "2024-01-01T00:00:00.000Z",
                "pattern": "[file:hashes.MD5 = 'd41d8cd98f00b204e9800998ecf8427e']",
                "labels": ["malicious-activity"]
            }
        ]
    }
    
    detection = parser.detect_data_type(stix_sample, "threat_intel.json")
    print("STIX 2.1 Detection:", detection)
    
    # Test adversarial numpy data
    adversarial_sample = np.random.rand(100, 32, 32, 3)  # Batch of images
    detection = parser.detect_data_type(adversarial_sample, "adversarial_examples.npy")
    print("Adversarial NumPy Detection:", detection) 