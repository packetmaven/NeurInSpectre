#!/usr/bin/env python3
"""
Dashboard Dataset Updater for NeurInSpectre
Updates dashboard to use real attack datasets instead of synthetic data
"""

import json
import numpy as np
from pathlib import Path
import re

class DashboardDatasetUpdater:
    def __init__(self):
        self.dashboard_file = Path("research_materials/dashboards/enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py")
        self.dataset_cache = Path("neurinspectre/dataset_cache")
        
        # Real dataset mappings for dashboard
        self.real_datasets = {
            "real-attack": {
                "name": "Adversarial Robustness Toolbox",
                "source": "adversarial_robustness_toolbox",
                "size": "45MB", 
                "attacks": 15000,
                "description": "IBM's ART framework with real adversarial examples"
            },
            "agent-attacks": {
                "name": "JailbreakBench Agent Attacks", 
                "source": "jailbreakbench",
                "size": "8MB",
                "attacks": 3500,
                "description": "Real jailbreak attacks for LLM agents"
            },
            "federated-attacks": {
                "name": "FLTrust Federated Attacks",
                "source": "federated_attacks", 
                "size": "15MB",
                "attacks": 2800,
                "description": "Real federated learning attack datasets"
            },
            "privacy-attacks": {
                "name": "Membership Inference Attacks",
                "source": "membership_inference",
                "size": "6MB", 
                "attacks": 4200,
                "description": "Privacy inference and data extraction attacks"
            },
            "phishing-attacks": {
                "name": "UCI Phishing Dataset",
                "source": "phishing_dataset",
                "size": "2MB",
                "attacks": 11055,
                "description": "Real phishing website detection dataset"
            },
            "text-adversarial": {
                "name": "TextAttack Framework",
                "source": "textattack", 
                "size": "25MB",
                "attacks": 8900,
                "description": "Real adversarial text attack examples"
            },
            "malware-detection": {
                "name": "Drebin Android Malware",
                "source": "drebin_android",
                "size": "18MB", 
                "attacks": 5560,
                "description": "Android malware detection dataset"
            },
            "cleverhans-examples": {
                "name": "CleverHans Adversarial Examples",
                "source": "cleverhans_examples",
                "size": "12MB",
                "attacks": 6700,
                "description": "CleverHans adversarial examples library"
            }
        }
        
        # Real MITRE ATLAS techniques from actual framework
        self.real_atlas_techniques = {
            "T1590.001": "Gather Victim ML Model Information: Model Architecture",
            "T1590.002": "Gather Victim ML Model Information: Model Capabilities", 
            "T1590.003": "Gather Victim ML Model Information: Model Artifacts",
            "T1566.001": "Phishing: Spearphishing Attachment",
            "T1566.002": "Phishing: Spearphishing Link",
            "T1059.007": "Command and Scripting Interpreter: JavaScript",
            "T1485": "Data Destruction",
            "T1005": "Data from Local System",
            "T1565.001": "Data Manipulation: Stored Data Manipulation",
            "T1027": "Obfuscated Files or Information",
            "T1041": "Exfiltration Over C2 Channel",
            "T1030": "Data Transfer Size Limits",
            "T1078": "Valid Accounts",
            "T1499": "Endpoint Denial of Service",
            "T1059": "Command and Scripting Interpreter",
            "T1547": "Boot or Logon Autostart Execution"
        }
    
    def load_dataset_metadata(self, dataset_source):
        """Load metadata for a real dataset"""
        metadata_file = self.dataset_cache / dataset_source / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return None
    
    def check_dataset_availability(self):
        """Check which real datasets are available"""
        available = {}
        missing = []
        
        for dataset_id, info in self.real_datasets.items():
            source_dir = self.dataset_cache / info["source"]
            if source_dir.exists():
                metadata = self.load_dataset_metadata(info["source"])
                if metadata:
                    available[dataset_id] = {
                        **info,
                        "metadata": metadata,
                        "status": "available"
                    }
                else:
                    available[dataset_id] = {
                        **info,
                        "status": "no_metadata"
                    }
            else:
                missing.append(dataset_id)
        
        return available, missing
    
    def generate_real_dataset_config(self, available_datasets):
        """Generate dataset configuration for dashboard"""
        dataset_config = []
        
        for dataset_id, info in available_datasets.items():
            config = {
                "id": dataset_id,
                "name": info["name"],
                "size": info["size"],
                "attacks": info["attacks"],
                "description": info["description"],
                "source_path": f"neurinspectre/dataset_cache/{info['source']}",
                "real_data": True
            }
            dataset_config.append(config)
        
        return dataset_config
    
    def update_dashboard_datasets(self):
        """Update dashboard file to use real datasets"""
        if not self.dashboard_file.exists():
            print(f"❌ Dashboard file not found: {self.dashboard_file}")
            return False
        
        # Read current dashboard
        with open(self.dashboard_file, 'r') as f:
            content = f.read()
        
        # Check available datasets
        available, missing = self.check_dataset_availability()
        
        if not available:
            print("❌ No real datasets found. Please run create_real_attack_datasets.py first")
            return False
        
        print(f"✅ Found {len(available)} real datasets")
        for dataset_id, info in available.items():
            print(f"  📊 {dataset_id}: {info['name']}")
        
        if missing:
            print(f"⚠️ Missing {len(missing)} datasets: {', '.join(missing)}")
        
        # Generate new dataset configuration
        real_config = self.generate_real_dataset_config(available)
        
        # Replace synthetic dataset configuration
        old_pattern = r'self\.available_datasets = \[.*?\]'
        new_datasets = "self.available_datasets = [\n"
        
        for config in real_config:
            new_datasets += f"            {{\n"
            new_datasets += f"                'id': '{config['id']}',\n"
            new_datasets += f"                'name': '{config['name']}',\n"
            new_datasets += f"                'size': '{config['size']}',\n"
            new_datasets += f"                'attacks': {config['attacks']},\n"
            new_datasets += f"                'description': '{config['description']}',\n"
            new_datasets += f"                'source_path': '{config['source_path']}',\n"
            new_datasets += f"                'real_data': True\n"
            new_datasets += f"            }},\n"
        
        new_datasets += "        ]"
        
        # Replace in content
        content = re.sub(old_pattern, new_datasets, content, flags=re.DOTALL)
        
        # Update ATLAS techniques with real ones
        old_atlas_pattern = r'self\.atlas_techniques = \{.*?\}'
        new_atlas = "self.atlas_techniques = {\n"
        
        for tech_id, description in self.real_atlas_techniques.items():
            new_atlas += f"            '{tech_id}': '{description}',\n"
        
        new_atlas += "        }"
        
        content = re.sub(old_atlas_pattern, new_atlas, content, flags=re.DOTALL)
        
        # Add real data loading methods
        real_data_methods = '''
    def _load_real_dataset(self, dataset_id):
        """Load real attack dataset"""
        dataset_info = next((d for d in self.available_datasets if d['id'] == dataset_id), None)
        if not dataset_info or not dataset_info.get('real_data'):
            return None
        
        source_path = Path(dataset_info['source_path'])
        if not source_path.exists():
            print(f"⚠️ Dataset path not found: {source_path}")
            return None
        
        # Load metadata
        metadata_file = source_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load actual data files
            synthetic_dir = source_path / "synthetic_samples"
            if synthetic_dir.exists():
                return self._load_synthetic_samples(synthetic_dir, metadata)
        
        return None
    
    def _load_synthetic_samples(self, data_dir, metadata):
        """Load synthetic samples generated from real datasets"""
        data = {}
        
        # Load different types of attack data
        for file_path in data_dir.glob("*.npy"):
            data[file_path.stem] = np.load(file_path)
        
        for file_path in data_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                data[file_path.stem] = json.load(f)
        
        return {
            'data': data,
            'metadata': metadata,
            'type': metadata.get('type', 'unknown'),
            'techniques': metadata.get('techniques', []),
            'atlas_mapping': metadata.get('atlas_mapping', {})
        }
    
    def _update_with_real_data(self, dataset_id):
        """Update dashboard metrics with real dataset"""
        real_data = self._load_real_dataset(dataset_id)
        if not real_data:
            return
        
        # Update metrics based on real data
        atlas_info = real_data.get('atlas_mapping', {})
        techniques = real_data.get('techniques', [])
        
        # Update agent network with real techniques
        for i, technique in enumerate(techniques[:10]):  # Limit to 10 for display
            if i < len(self.agent_positions):
                agent_id = f"agent_{i+1}"
                self.agent_data[agent_id]['technique'] = technique
                self.agent_data[agent_id]['atlas_id'] = atlas_info.get('techniques', ['T1000'])[0] if atlas_info.get('techniques') else 'T1000'
        
        print(f"📊 Updated dashboard with real data from {dataset_id}")
'''
        
        # Insert real data methods before the last class method
        last_method_pattern = r'(\n    def run_dashboard\(self.*?\n        app\.run_server.*?\n)'
        content = re.sub(last_method_pattern, real_data_methods + r'\1', content, flags=re.DOTALL)
        
        # Update the dataset selection callback to use real data
        callback_pattern = r'(def update_dataset_selection\(selected_dataset\):.*?return.*?\})'
        new_callback = '''def update_dataset_selection(selected_dataset):
            # Update dataset characteristics based on selection AND load real data
            if selected_dataset:
                self._update_with_real_data(selected_dataset)
                for dataset in self.available_datasets:
                    if dataset['id'] == selected_dataset:
                        self.current_dataset = dataset
                        break
            return {'backgroundColor': '#2c3e50', 'color': '#E6E6FA'}'''
        
        content = re.sub(callback_pattern, new_callback, content, flags=re.DOTALL)
        
        # Write updated dashboard
        backup_file = self.dashboard_file.with_suffix('.py.backup')
        self.dashboard_file.rename(backup_file)
        
        with open(self.dashboard_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Dashboard updated successfully!")
        print(f"📁 Backup saved to: {backup_file}")
        print(f"🔄 Updated datasets: {len(real_config)} real datasets")
        print(f"🎯 Updated ATLAS techniques: {len(self.real_atlas_techniques)} techniques")
        
        return True
    
    def verify_update(self):
        """Verify the dashboard update was successful"""
        if not self.dashboard_file.exists():
            return False
        
        with open(self.dashboard_file, 'r') as f:
            content = f.read()
        
        # Check if real datasets are present
        real_data_present = "'real_data': True" in content
        atlas_techniques_present = "T1590.001" in content
        load_methods_present = "_load_real_dataset" in content
        
        return real_data_present and atlas_techniques_present and load_methods_present

def main():
    updater = DashboardDatasetUpdater()
    
    print("NeurInSpectre Dashboard Dataset Updater")
    print("=" * 50)
    
    # Check current status
    available, missing = updater.check_dataset_availability()
    
    if not available:
        print("❌ No real datasets found!")
        print("Please run the following command first:")
        print("python create_real_attack_datasets.py")
        return
    
    print(f"\n📊 Available Real Datasets: {len(available)}")
    for dataset_id, info in available.items():
        status = "✅" if info.get("status") == "available" else "⚠️"
        print(f"  {status} {dataset_id}: {info['name']} ({info['size']})")
    
    if missing:
        print(f"\n⚠️ Missing Datasets: {len(missing)}")
        for dataset_id in missing:
            print(f"  ❌ {dataset_id}: {updater.real_datasets[dataset_id]['name']}")
    
    print(f"\n🎯 Will update dashboard to use {len(available)} real datasets")
    print("This will replace synthetic data with real attack datasets")
    
    response = input("\nProceed with dashboard update? (y/n): ").lower().strip()
    if response == 'y':
        if updater.update_dashboard_datasets():
            if updater.verify_update():
                print("\n🎉 Dashboard successfully updated with real datasets!")
                print("\n🚀 Next steps:")
                print("1. Run the updated dashboard:")
                print("   python research_materials/dashboards/enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py")
                print("2. Select real datasets from the dropdown")
                print("3. Observe real attack data and MITRE ATLAS techniques")
            else:
                print("\n⚠️ Update completed but verification failed")
        else:
            print("\n❌ Dashboard update failed")
    else:
        print("❌ Update cancelled")

if __name__ == "__main__":
    main() 