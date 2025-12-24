#!/bin/bash
# Backup Verification Script - June 23, 2025

echo "🔍 Verifying NeurInSpectre Dashboard Backups..."
echo "================================================"

# Check backup directories exist
echo "📁 Checking backup directories..."
if [ -d "Backup_Dashboard1_20250623" ]; then
    echo "✅ Dashboard 1 backup directory exists"
else
    echo "❌ Dashboard 1 backup directory missing"
fi

if [ -d "Backup_Dashboard2_20250623" ]; then
    echo "✅ Dashboard 2 backup directory exists"
else
    echo "❌ Dashboard 2 backup directory missing"
fi

echo ""
echo "📄 Checking backup files..."

# Check Dashboard 1 files
echo "🔍 Dashboard 1 (TTD - Port 8080):"
if [ -f "Backup_Dashboard1_20250623/ttd.py" ]; then
    echo "  ✅ ttd.py ($(wc -l < Backup_Dashboard1_20250623/ttd.py) lines)"
else
    echo "  ❌ ttd.py missing"
fi

if [ -f "Backup_Dashboard1_20250623/__init__.py" ]; then
    echo "  ✅ __init__.py"
else
    echo "  ❌ __init__.py missing"
fi

if [ -f "Backup_Dashboard1_20250623/RESTORE_INSTRUCTIONS.md" ]; then
    echo "  ✅ RESTORE_INSTRUCTIONS.md"
else
    echo "  ❌ RESTORE_INSTRUCTIONS.md missing"
fi

# Check Dashboard 2 files
echo ""
echo "🔍 Dashboard 2 (MPS ATLAS Agent - Port 8117):"
if [ -f "Backup_Dashboard2_20250623/enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py" ]; then
    echo "  ✅ enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py ($(wc -l < Backup_Dashboard2_20250623/enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py) lines)"
else
    echo "  ❌ enhanced_mps_atlas_agent_dashboard_COMPLETE_WORKING.py missing"
fi

if [ -f "Backup_Dashboard2_20250623/RESTORE_INSTRUCTIONS.md" ]; then
    echo "  ✅ RESTORE_INSTRUCTIONS.md"
else
    echo "  ❌ RESTORE_INSTRUCTIONS.md missing"
fi

echo ""
echo "🌐 Checking if dashboards are currently running..."
if curl -s -I http://127.0.0.1:8080 > /dev/null 2>&1; then
    echo "✅ Dashboard 1 (Port 8080) is accessible"
else
    echo "❌ Dashboard 1 (Port 8080) not accessible"
fi

if curl -s -I http://127.0.0.1:8117 > /dev/null 2>&1; then
    echo "✅ Dashboard 2 (Port 8117) is accessible"
else
    echo "❌ Dashboard 2 (Port 8117) not accessible"
fi

echo ""
echo "📊 Backup Summary:"
echo "  Dashboard 1: TTD (Time to Detection) - Port 8080"
echo "  Dashboard 2: MPS ATLAS Agent - Port 8117"
echo "  Backup Date: June 23, 2025"
echo "  Location: $(pwd)"
echo ""
echo "🔗 Access URLs:"
echo "  Dashboard 1: http://127.0.0.1:8080"
echo "  Dashboard 2: http://127.0.0.1:8117"
echo ""
echo "✅ Backup verification complete!"
