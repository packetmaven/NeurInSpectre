#!/usr/bin/env python3
"""
CLI Data Upload System for NeurInSpectre
Comprehensive command-line interface for uploading and testing industry-standard data formats
"""

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

try:
    _upload_mod = importlib.import_module("enhanced_data_upload_system")
    IndustryStandardDataParser = getattr(_upload_mod, "IndustryStandardDataParser")
    parse_uploaded_file_enhanced = getattr(_upload_mod, "parse_uploaded_file_enhanced")
    generate_upload_status_display = getattr(_upload_mod, "generate_upload_status_display")
    UPLOAD_SYSTEM_AVAILABLE = True
except Exception:
    UPLOAD_SYSTEM_AVAILABLE = False
    IndustryStandardDataParser = None  # type: ignore
    parse_uploaded_file_enhanced = None  # type: ignore
    generate_upload_status_display = None  # type: ignore

logger = logging.getLogger(__name__)

if not UPLOAD_SYSTEM_AVAILABLE:
    # Keep import side-effects minimal: no prints. CLI entrypoints can surface guidance.
    logger.debug("Enhanced upload system not available (optional dependency missing).")

class DataUploadCLI:
    """
    Command-line interface for data upload and analysis
    """
    
    def __init__(self):
        self.parser = IndustryStandardDataParser() if UPLOAD_SYSTEM_AVAILABLE else None
        self.upload_history = []
        self.results_dir = Path("upload_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def upload_file(self, file_path: str, output_format: str = "json") -> Dict[str, Any]:
        """
        Upload and analyze a single file
        """
        if not UPLOAD_SYSTEM_AVAILABLE:
            return {"error": "Upload system not available"}
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        print(f"📁 Uploading file: {file_path.name}")
        print(f"📊 File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Create base64 encoded content (simulating web upload)
            import base64
            encoded_content = base64.b64encode(file_content).decode('utf-8')
            contents = f"data:application/octet-stream;base64,{encoded_content}"
            
            # Parse the file
            parsed_data, format_type, metadata = parse_uploaded_file_enhanced(
                contents, file_path.name
            )
            
            # Generate status display
            status_info = generate_upload_status_display(
                file_path.name, format_type, metadata
            )
            
            # Create result
            result = {
                "timestamp": datetime.now().isoformat(),
                "filename": file_path.name,
                "file_size_mb": file_path.stat().st_size / 1024 / 1024,
                "format_type": format_type,
                "status_info": status_info,
                "metadata": metadata,
                "success": parsed_data is not None
            }
            
            # Add to history
            self.upload_history.append(result)
            
            # Display results
            self.display_result(result)
            
            # Save results if requested
            if output_format in ["json", "all"]:
                self.save_result(result, file_path.name)
            
            return result
            
        except Exception as e:
            error_result = {
                "timestamp": datetime.now().isoformat(),
                "filename": file_path.name,
                "error": str(e),
                "success": False
            }
            
            print(f"❌ Error processing {file_path.name}: {e}")
            return error_result
    
    def upload_directory(self, directory_path: str, pattern: str = "*") -> List[Dict[str, Any]]:
        """
        Upload all files in a directory matching the pattern
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            print(f"❌ Directory not found: {directory_path}")
            return []
        
        # Find matching files
        files = list(directory_path.glob(pattern))
        
        if not files:
            print(f"❌ No files found matching pattern: {pattern}")
            return []
        
        print(f"📁 Found {len(files)} files to upload")
        print("=" * 60)
        
        results = []
        for file_path in files:
            if file_path.is_file():
                result = self.upload_file(str(file_path))
                results.append(result)
                print("-" * 40)
        
        # Summary
        successful = sum(1 for r in results if r.get("success", False))
        print("\n📊 Upload Summary:")
        print(f"   Total files: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        
        return results
    
    def display_result(self, result: Dict[str, Any]):
        """
        Display upload result in a formatted way
        """
        if not result.get("success", False):
            print(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
            return
        
        status_info = result.get("status_info", {})
        metadata = result.get("metadata", {})
        
        print("✅ Upload successful!")
        print(f"   📄 Format: {result.get('format_type', 'unknown')}")
        print(f"   📊 Category: {status_info.get('data_category', 'unknown')}")
        print(f"   🎯 Attack Data: {'Yes' if status_info.get('is_attack_data', False) else 'No'}")
        print(f"   📈 Confidence: {status_info.get('confidence', 0):.1%}")
        
        # Display key metadata
        if metadata:
            print("   📋 Metadata:")
            for key, value in list(metadata.items())[:5]:  # Show first 5 items
                if isinstance(value, (int, float, str)):
                    print(f"      {key}: {value}")
                elif isinstance(value, list) and len(value) < 10:
                    print(f"      {key}: {', '.join(map(str, value))}")
    
    def save_result(self, result: Dict[str, Any], filename: str):
        """
        Save upload result to file
        """
        result_file = self.results_dir / f"{filename}_result.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"💾 Result saved to: {result_file}")
    
    def list_supported_formats(self):
        """
        List all supported formats
        """
        formats = {
            "json": "STIX 2.1, MITRE ATT&CK, General JSON",
            "csv": "Threat Intelligence, IoCs, Tabular Data",
            "xml": "STIX 1.x, Legacy XML Formats",
            "yaml": "Security Rules, Configuration Files",
            "yml": "Security Rules, Configuration Files",
            "png": "Images (metadata-only parsing; safe, no execution)",
            "npy": "Adversarial Examples, Gradients, NumPy Arrays",
            "npz": "Multi-array Adversarial Data, Compressed NumPy",
            "h5": "Large-scale ML Datasets, Scientific Data",
            "hdf5": "HDF5 Scientific Datasets",
            "pkl": "Python Objects, ML Models",
            "pickle": "Serialized Python Data"
        }
        
        print("🔒 Supported Industry-Standard Formats:")
        print("=" * 60)
        
        for ext, desc in formats.items():
            print(f"   .{ext:<8} - {desc}")
        
        print("\n📋 Standards Compliance:")
        print("   • OASIS STIX 2.1 (Threat Intelligence)")
        print("   • MITRE ATT&CK Framework")
        print("   • NIST AI 100-2 (2025) Adversarial ML")
        print("   • IEEE HDF5 Scientific Data")
        print("   • YAML Security Rules")
    
    def show_upload_history(self):
        """
        Show upload history
        """
        if not self.upload_history:
            print("📋 No upload history available")
            return
        
        print(f"📋 Upload History ({len(self.upload_history)} files):")
        print("=" * 80)
        
        for i, result in enumerate(self.upload_history, 1):
            status = "✅" if result.get("success", False) else "❌"
            filename = result.get("filename", "unknown")
            format_type = result.get("format_type", "unknown")
            timestamp = result.get("timestamp", "unknown")
            
            print(f"{i:2d}. {status} {filename:<30} {format_type:<20} {timestamp}")
    
    def test_all_formats(self):
        """
        Test all supported formats using sample files
        """
        sample_dir = Path("sample_upload_test_files")
        
        if not sample_dir.exists():
            print("❌ Sample files not found. Generate them first:")
            print("   python test_enhanced_upload_formats.py")
            return
        
        print("🧪 Testing all supported formats...")
        print("=" * 60)
        
        results = self.upload_directory(str(sample_dir))
        
        # Analyze results by format
        format_results = {}
        for result in results:
            if result.get("success", False):
                format_type = result.get("format_type", "unknown")
                if format_type not in format_results:
                    format_results[format_type] = []
                format_results[format_type].append(result)
        
        print("\n📊 Format Analysis:")
        print("=" * 60)
        
        for format_type, format_results_list in format_results.items():
            avg_confidence = sum(
                r.get("status_info", {}).get("confidence", 0) 
                for r in format_results_list
            ) / len(format_results_list)
            
            attack_data_count = sum(
                1 for r in format_results_list 
                if r.get("status_info", {}).get("is_attack_data", False)
            )
            
            print(f"   {format_type}:")
            print(f"     Files: {len(format_results_list)}")
            print(f"     Avg Confidence: {avg_confidence:.1%}")
            print(f"     Attack Data: {attack_data_count}/{len(format_results_list)}")

def main():
    """
    Main CLI function
    """
    parser = argparse.ArgumentParser(
        description="NeurInSpectre Data Upload CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a single file
  python -m neurinspectre.cli.data_upload_cli upload file.json
  
  # Upload all files in a directory
  python -m neurinspectre.cli.data_upload_cli upload-dir sample_files/
  
  # Upload specific file types
  python -m neurinspectre.cli.data_upload_cli upload-dir sample_files/ "*.json"
  
  # Test all supported formats
  python -m neurinspectre.cli.data_upload_cli test-all
  
  # Show supported formats
  python -m neurinspectre.cli.data_upload_cli formats
  
  # Show upload history
  python -m neurinspectre.cli.data_upload_cli history
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload single file
    upload_parser = subparsers.add_parser('upload', help='Upload a single file')
    upload_parser.add_argument('file', help='Path to file to upload')
    upload_parser.add_argument('--output', choices=['json', 'none'], default='json',
                              help='Output format for results')
    
    # Upload directory
    upload_dir_parser = subparsers.add_parser('upload-dir', help='Upload all files in directory')
    upload_dir_parser.add_argument('directory', help='Path to directory')
    upload_dir_parser.add_argument('pattern', nargs='?', default='*',
                                  help='File pattern to match (default: *)')
    
    # Test all formats
    subparsers.add_parser('test-all', help='Test all supported formats')
    
    # Show formats
    subparsers.add_parser('formats', help='List supported formats')
    
    # Show history
    subparsers.add_parser('history', help='Show upload history')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Check if upload system is available
    if not UPLOAD_SYSTEM_AVAILABLE and args.command in ['upload', 'upload-dir', 'test-all']:
        print("❌ Enhanced upload system not available")
        print("📦 Install dependencies: pip install pyyaml h5py lxml")
        return
    
    # Initialize CLI
    cli = DataUploadCLI()
    
    # Execute command
    if args.command == 'upload':
        cli.upload_file(args.file, args.output)
    
    elif args.command == 'upload-dir':
        cli.upload_directory(args.directory, args.pattern)
    
    elif args.command == 'test-all':
        cli.test_all_formats()
    
    elif args.command == 'formats':
        cli.list_supported_formats()
    
    elif args.command == 'history':
        cli.show_upload_history()

if __name__ == "__main__":
    main() 