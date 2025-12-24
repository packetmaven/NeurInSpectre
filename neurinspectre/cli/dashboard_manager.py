#!/usr/bin/env python3
"""
NeurInSpectre Dashboard Management CLI
Integrated backup, restoration, and management for all dashboards
"""

import os
import sys
import json
import time
import shutil
import psutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import tempfile

class DashboardManager:
    """Comprehensive dashboard management system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.backup_dir = self.project_root / "dashboard_backups"
        self.backup_dir.mkdir(exist_ok=True)
        # Logs directory (writable by default). Override with NEURINSPECTRE_LOG_DIR
        default_logs_dir = Path(tempfile.gettempdir()) / "neurinspectre_logs"
        env_logs_dir = os.environ.get("NEURINSPECTRE_LOG_DIR")
        self.logs_dir = Path(env_logs_dir) if env_logs_dir else default_logs_dir
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Fallback: try project_root/logs (may be read-only in some environments)
            self.logs_dir = self.project_root / "logs"
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Dashboard configuration
        #
        # IMPORTANT:
        # - Use sys.executable so we always start dashboards under the same Python environment
        #   that is running `neurinspectre dashboard-manager ...` (avoids relying on `python` being on PATH).
        # - Keep file paths aligned to the repo layout (some legacy research_materials scripts are optional).
        py = sys.executable
        self.dashboards = {
            'ttd': {
                'name': 'TTD Dashboard (Time Travel Debugger)',
                'port': 8080,
                'file': 'neurinspectre/cli/ttd.py',
                'command': [py, '-m', 'neurinspectre.cli', 'dashboard', '--port', '8080', '--host', '127.0.0.1', '--debug'],
                'log': 'ttd_dash8080.log'
            },
            '3x2': {
                'name': 'Enhanced 3×2 Layout Dashboard',
                'port': 8154,
                'file': 'research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final_3x2.py',
                'command': [py, 'research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final_3x2.py', '--port', '8154'],
                'log': 'dash8154.log'
            },
            'intelligence': {
                'name': 'Actionable Intelligence Dashboard',
                'port': 8155,
                'file': 'research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final.py',
                'command': [py, 'research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final.py', '--port', '8155'],
                'log': 'dash8155.log'
            },
            'research': {
                'name': '2025 AI Security Research Dashboard',
                'port': 8156,
                'file': 'neurinspectre/ai_security_research_dashboard_2025.py',
                'command': [py, '-m', 'neurinspectre.cli', 'ai-security-dashboard', '--port', '8156', '--host', '127.0.0.1'],
                'log': 'dash8156.log'
            },
            'enhanced': {
                'name': 'Enhanced Intelligence Dashboard',
                'port': 8152,
                'file': 'research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final_working.py',
                'command': [py, 'research_materials/dashboards/absolutely_fixed_dropdown_dashboard_enhanced_final_working.py', '--port', '8152'],
                'log': 'dash8152.log'
            }
        }
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a complete backup of all dashboards"""
        if not backup_name:
            backup_name = f"dashboard_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)
        
        print(f"🛡️ Creating dashboard backup: {backup_name}")
        print(f"📁 Backup location: {backup_path}")
        
        # Backup metadata
        metadata = {
            'backup_name': backup_name,
            'created': datetime.now().isoformat(),
            'dashboards': {},
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'cwd': str(self.project_root)
            }
        }
        
        # Backup each dashboard file
        for dash_id, config in self.dashboards.items():
            source_file = self.project_root / config['file']
            
            if source_file.exists():
                backup_file = backup_path / f"{dash_id}_backup.py"
                shutil.copy2(source_file, backup_file)
                
                metadata['dashboards'][dash_id] = {
                    'name': config['name'],
                    'port': config['port'],
                    'original_file': str(source_file),
                    'backup_file': str(backup_file),
                    'file_size': source_file.stat().st_size,
                    'backed_up': True
                }
                
                print(f"   ✅ Backed up {config['name']} ({source_file.stat().st_size} bytes)")
            else:
                print(f"   ⚠️  File not found: {source_file}")
                metadata['dashboards'][dash_id] = {
                    'name': config['name'],
                    'port': config['port'],
                    'original_file': str(source_file),
                    'backed_up': False,
                    'error': 'File not found'
                }
        
        # Save metadata
        with open(backup_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create restoration script
        self._create_restoration_script(backup_path, backup_name)
        
        print(f"✅ Backup created successfully: {backup_name}")
        return backup_name
    
    def _create_restoration_script(self, backup_path: Path, backup_name: str):
        """Create a restoration script for the backup"""
        script_content = f'''#!/usr/bin/env python3
"""
Auto-generated restoration script for backup: {backup_name}
"""

import subprocess
import sys
import os

def restore_backup():
    """Restore dashboards from backup"""
    print("🔄 Restoring dashboards from backup: {backup_name}")
    
    # Add project root to Python path
    project_root = "{self.project_root}"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import dashboard manager
    from neurinspectre.cli.dashboard_manager import DashboardManager
    
    # Create manager and restore
    manager = DashboardManager()
    success = manager.restore_backup("{backup_name}")
    
    if success:
        print("✅ Restoration completed successfully!")
        return 0
    else:
        print("❌ Restoration failed!")
        return 1

if __name__ == "__main__":
    sys.exit(restore_backup())
'''
        
        script_path = backup_path / 'restore.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        script_path.chmod(0o755)
        print(f"   📜 Created restoration script: {script_path}")
    
    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                metadata_file = backup_dir / 'metadata.json'
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        backups.append(metadata)
                    except Exception as e:
                        print(f"⚠️  Error reading backup metadata: {e}")
        
        return sorted(backups, key=lambda x: x['created'], reverse=True)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore dashboards from a specific backup"""
        backup_path = self.backup_dir / backup_name
        metadata_file = backup_path / 'metadata.json'
        
        if not metadata_file.exists():
            print(f"❌ Backup not found: {backup_name}")
            return False
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"🔄 Restoring backup: {backup_name}")
        print(f"📅 Created: {metadata['created']}")
        
        # Stop all dashboards first
        print("🛑 Stopping existing dashboards...")
        self.stop_all_dashboards()
        
        # Restore each dashboard file
        restored_count = 0
        for dash_id, backup_info in metadata['dashboards'].items():
            if backup_info.get('backed_up', False):
                backup_file = Path(backup_info['backup_file'])
                original_file = Path(backup_info['original_file'])
                
                if backup_file.exists():
                    # Ensure target directory exists
                    original_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Restore file
                    shutil.copy2(backup_file, original_file)
                    print(f"   ✅ Restored {backup_info['name']}")
                    restored_count += 1
                else:
                    print(f"   ❌ Backup file not found: {backup_file}")
            else:
                print(f"   ⚠️  Skipped {backup_info['name']} (not backed up)")
        
        if restored_count > 0:
            print(f"✅ Restored {restored_count} dashboards")
            
            # Optionally start dashboards
            print("🚀 Starting restored dashboards...")
            self.start_all_dashboards()
            return True
        else:
            print("❌ No dashboards were restored")
            return False
    
    def get_dashboard_status(self) -> Dict:
        """Get status of all dashboards"""
        status = {}
        
        for dash_id, config in self.dashboards.items():
            port = config['port']
            status[dash_id] = {
                'name': config['name'],
                'port': port,
                'process_running': self._is_process_running(port),
                'port_listening': self._is_port_listening(port),
                'http_accessible': self._is_http_accessible(port),
                'log_file': str(self.logs_dir / config['log'])
            }
        
        return status
    
    def _is_process_running(self, port: int) -> bool:
        """Check if a process is running on the specified port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                cmdline = proc.info['cmdline']
                if cmdline and any(str(port) in arg for arg in cmdline):
                    return True
            return False
        except Exception:
            return False
    
    def _is_port_listening(self, port: int) -> bool:
        """Check if port is listening"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return True
        except Exception:
            pass
        # Fallback: try opening a socket connection
        try:
            import socket
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except Exception:
            return False
    
    def _is_http_accessible(self, port: int) -> bool:
        """Check if HTTP endpoint is accessible"""
        try:
            import urllib.request
            urllib.request.urlopen(f'http://127.0.0.1:{port}', timeout=5)
            return True
        except Exception:
            return False
    
    def start_dashboard(self, dashboard_id: str) -> bool:
        """Start a specific dashboard"""
        if dashboard_id not in self.dashboards:
            print(f"❌ Unknown dashboard: {dashboard_id}")
            return False
        
        config = self.dashboards[dashboard_id]
        
        # Check if already running
        if self._is_port_listening(config['port']):
            print(f"⚠️  Dashboard already running on port {config['port']}")
            return True
        
        print(f"🚀 Starting {config['name']} on port {config['port']}...")
        
        try:
            # Start process; write logs to a guaranteed-writable directory
            log_path = self.logs_dir / config['log']
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = open(log_path, 'w')
            process = subprocess.Popen(
                config['command'],
                cwd=self.project_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )
            
            # Wait a moment for startup
            time.sleep(3)
            
            # Check if it started successfully (accept HTTP accessible or port listening)
            listening = self._is_port_listening(config['port'])
            http_ok = self._is_http_accessible(config['port'])
            if process.poll() is None and (listening or http_ok):
                print(f"   ✅ Started successfully (PID: {process.pid})")
                return True
            else:
                print(f"   ❌ Failed to start")
                return False
                
        except Exception as e:
            print(f"   ❌ Error starting dashboard: {e}")
            return False
    
    def stop_dashboard(self, dashboard_id: str) -> bool:
        """Stop a specific dashboard"""
        if dashboard_id not in self.dashboards:
            print(f"❌ Unknown dashboard: {dashboard_id}")
            return False
        
        config = self.dashboards[dashboard_id]
        port = config['port']
        
        print(f"🛑 Stopping {config['name']} (port {port})...")
        
        try:
            # Find and kill processes using this port
            killed_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any(str(port) in arg for arg in cmdline):
                        proc.terminate()
                        killed_count += 1
                        print(f"   ✅ Terminated process {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Wait for processes to stop
            time.sleep(2)
            
            # Force kill if still running
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and any(str(port) in arg for arg in cmdline):
                        proc.kill()
                        print(f"   ⚡ Force killed process {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return killed_count > 0
            
        except Exception as e:
            print(f"   ❌ Error stopping dashboard: {e}")
            return False
    
    def start_all_dashboards(self) -> int:
        """Start all dashboards"""
        print("🚀 Starting all dashboards...")
        
        started_count = 0
        for dash_id in self.dashboards:
            if self.start_dashboard(dash_id):
                started_count += 1
            time.sleep(2)  # Stagger starts
        
        print(f"✅ Started {started_count}/{len(self.dashboards)} dashboards")
        return started_count
    
    def stop_all_dashboards(self) -> int:
        """Stop all dashboards"""
        print("🛑 Stopping all dashboards...")
        
        stopped_count = 0
        for dash_id in self.dashboards:
            if self.stop_dashboard(dash_id):
                stopped_count += 1
        
        print(f"✅ Stopped {stopped_count} dashboards")
        return stopped_count
    
    def restart_all_dashboards(self) -> int:
        """Restart all dashboards"""
        print("🔄 Restarting all dashboards...")
        
        self.stop_all_dashboards()
        time.sleep(5)
        return self.start_all_dashboards()
    
    def emergency_restore(self, backup_name: Optional[str] = None) -> bool:
        """Emergency restoration - kill everything and restore"""
        print("🚨 EMERGENCY DASHBOARD RESTORATION")
        print("=" * 50)
        
        # Force kill all dashboard processes
        print("🛑 Force killing all dashboard processes...")
        try:
            subprocess.run(['pkill', '-9', '-f', 'dashboard'], capture_output=True)
            subprocess.run(['pkill', '-9', '-f', 'ttd'], capture_output=True)
            for port in [8080, 8154, 8155, 8156, 8152]:
                subprocess.run(['pkill', '-9', '-f', f'python.*{port}'], capture_output=True)
        except Exception as e:
            print(f"   ⚠️  Error during force kill: {e}")
        
        time.sleep(5)
        
        # Use latest backup if none specified
        if not backup_name:
            backups = self.list_backups()
            if not backups:
                print("❌ No backups available for emergency restoration!")
                return False
            backup_name = backups[0]['backup_name']
            print(f"📁 Using latest backup: {backup_name}")
        
        # Restore from backup
        success = self.restore_backup(backup_name)
        
        if success:
            print("🎉 EMERGENCY RESTORATION COMPLETED!")
        else:
            print("❌ EMERGENCY RESTORATION FAILED!")
        
        return success

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='NeurInSpectre Dashboard Management CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create backup
  python -m neurinspectre.cli dashboard-manager backup --name my_backup
  
  # List backups
  python -m neurinspectre.cli dashboard-manager list
  
  # Restore from backup
  python -m neurinspectre.cli dashboard-manager restore --name my_backup
  
  # Check status
  python -m neurinspectre.cli dashboard-manager status
  
  # Start all dashboards
  python -m neurinspectre.cli dashboard-manager start --all
  
  # Stop specific dashboard
  python -m neurinspectre.cli dashboard-manager stop --dashboard ttd
  
  # Emergency restore
  python -m neurinspectre.cli dashboard-manager emergency
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create dashboard backup')
    backup_parser.add_argument('--name', help='Backup name (default: auto-generated)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available backups')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('--name', required=True, help='Backup name to restore')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check dashboard status')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start dashboards')
    start_group = start_parser.add_mutually_exclusive_group(required=True)
    start_group.add_argument('--all', action='store_true', help='Start all dashboards')
    start_group.add_argument('--dashboard', help='Start specific dashboard')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop dashboards')
    stop_group = stop_parser.add_mutually_exclusive_group(required=True)
    stop_group.add_argument('--all', action='store_true', help='Stop all dashboards')
    stop_group.add_argument('--dashboard', help='Stop specific dashboard')
    
    # Restart command
    restart_parser = subparsers.add_parser('restart', help='Restart all dashboards')
    
    # Emergency command
    emergency_parser = subparsers.add_parser('emergency', help='Emergency restoration')
    emergency_parser.add_argument('--backup', help='Specific backup to use (default: latest)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create dashboard manager
    manager = DashboardManager()
    
    try:
        if args.command == 'backup':
            backup_name = manager.create_backup(args.name)
            print(f"🎉 Backup created: {backup_name}")
            return 0
            
        elif args.command == 'list':
            backups = manager.list_backups()
            if not backups:
                print("📭 No backups found")
                return 0
            
            print("📋 Available backups:")
            for backup in backups:
                print(f"   📁 {backup['backup_name']}")
                print(f"      📅 Created: {backup['created']}")
                print(f"      📊 Dashboards: {len(backup['dashboards'])}")
                print()
            return 0
            
        elif args.command == 'restore':
            success = manager.restore_backup(args.name)
            return 0 if success else 1
            
        elif args.command == 'status':
            status = manager.get_dashboard_status()
            print("📊 Dashboard Status:")
            print("=" * 50)
            
            for dash_id, info in status.items():
                print(f"🔍 {info['name']} (Port {info['port']}):")
                print(f"   Process: {'✅ Running' if info['process_running'] else '❌ Not running'}")
                print(f"   Port: {'✅ Listening' if info['port_listening'] else '❌ Not listening'}")
                print(f"   HTTP: {'✅ Accessible' if info['http_accessible'] else '❌ Not accessible'}")
                print(f"   Log: {info['log_file']}")
                print()
            return 0
            
        elif args.command == 'start':
            if args.all:
                count = manager.start_all_dashboards()
                return 0 if count > 0 else 1
            else:
                success = manager.start_dashboard(args.dashboard)
                return 0 if success else 1
                
        elif args.command == 'stop':
            if args.all:
                count = manager.stop_all_dashboards()
                return 0 if count >= 0 else 1
            else:
                success = manager.stop_dashboard(args.dashboard)
                return 0 if success else 1
                
        elif args.command == 'restart':
            count = manager.restart_all_dashboards()
            return 0 if count > 0 else 1
            
        elif args.command == 'emergency':
            success = manager.emergency_restore(args.backup)
            return 0 if success else 1
            
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 