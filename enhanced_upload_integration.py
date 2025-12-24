#!/usr/bin/env python3
"""
Enhanced Upload Integration for NeurInSpectre Dashboard
Integrates industry-standard data format support into existing dashboards
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import base64
import io
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime

# Import our enhanced parser
from enhanced_data_upload_system import (
    parse_uploaded_file_enhanced, 
    generate_upload_status_display,
    IndustryStandardDataParser
)

logger = logging.getLogger(__name__)

class EnhancedUploadComponent:
    """
    Enhanced upload component supporting all industry-standard formats
    """
    
    def __init__(self):
        self.supported_formats = [
            'json', 'csv', 'xml', 'yaml', 'yml', 
            'png',
            'npy', 'npz', 'h5', 'hdf5', 'pkl', 'pickle'
        ]
        
        self.format_descriptions = {
            'json': 'STIX 2.1, MITRE ATT&CK, General JSON',
            'csv': 'Tabular Threat Intelligence, IoCs',
            'xml': 'STIX 1.x, Legacy XML formats',
            'yaml': 'Security Rules, Configuration',
            'png': 'Image upload (metadata-only parsing)',
            'npy': 'Adversarial Examples, Gradients',
            'npz': 'Multi-array Adversarial Data',
            'h5': 'Large-scale ML Datasets',
            'hdf5': 'HDF5 Scientific Data',
            'pkl': 'Python Objects, Models',
            'pickle': 'Serialized Python Data'
        }
    
    def create_upload_component(self) -> dbc.Card:
        """Create the enhanced upload component"""
        
        # Format information cards
        format_cards = []
        for fmt, desc in self.format_descriptions.items():
            format_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(f".{fmt}", className="card-title text-primary"),
                            html.P(desc, className="card-text small")
                        ])
                    ], className="mb-2", style={"height": "80px"})
                ], width=3)
            )
        
        upload_component = dbc.Card([
            dbc.CardHeader([
                html.H4("🔒 Industry-Standard Data Upload", className="mb-0"),
                html.Small("Supports STIX 2.1, MITRE ATT&CK, NIST AI, and ML formats", 
                          className="text-muted")
            ]),
            dbc.CardBody([
                # Upload area
                dcc.Upload(
                    id='enhanced-upload',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt fa-3x mb-3"),
                        html.H5("Drag & Drop or Click to Upload"),
                        html.P("Supports all industry-standard cybersecurity formats", 
                               className="text-muted"),
                        html.Hr(),
                        html.P("Max file size: 500MB", className="small text-muted")
                    ]),
                    style={
                        'width': '100%',
                        'height': '150px',
                        'lineHeight': '150px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'textAlign': 'center',
                        'margin': '10px',
                        'borderColor': '#007bff'
                    },
                    multiple=True,
                    max_size=500*1024*1024  # 500MB
                ),
                
                # Format support information
                html.Div([
                    html.H6("📋 Supported Formats", className="mt-4 mb-3"),
                    dbc.Row(format_cards)
                ]),
                
                # Upload status area
                html.Div(id='upload-status', className="mt-4"),
                
                # Data preview area
                html.Div(id='data-preview', className="mt-4")
            ])
        ])
        
        return upload_component
    
    def create_upload_callbacks(self, app):
        """Create callbacks for the enhanced upload component"""
        
        @app.callback(
            [Output('upload-status', 'children'),
             Output('data-preview', 'children')],
            [Input('enhanced-upload', 'contents')],
            [State('enhanced-upload', 'filename'),
             State('enhanced-upload', 'last_modified')]
        )
        def handle_upload(contents_list, filename_list, last_modified_list):
            if not contents_list:
                return "", ""
            
            upload_status_cards = []
            preview_components = []
            
            for contents, filename, last_modified in zip(contents_list, filename_list, last_modified_list):
                try:
                    # Parse the uploaded file
                    parsed_data, format_type, metadata = parse_uploaded_file_enhanced(contents, filename)
                    
                    # Generate status display
                    status_info = generate_upload_status_display(filename, format_type, metadata)
                    
                    # Create status card
                    status_card = self.create_status_card(status_info, last_modified)
                    upload_status_cards.append(status_card)
                    
                    # Create preview component
                    if parsed_data is not None:
                        preview_component = self.create_preview_component(
                            parsed_data, status_info, filename
                        )
                        preview_components.append(preview_component)
                    
                except Exception as e:
                    logger.error(f"Error processing upload {filename}: {e}")
                    error_card = self.create_error_card(filename, str(e))
                    upload_status_cards.append(error_card)
            
            return upload_status_cards, preview_components
    
    def create_status_card(self, status_info: Dict[str, Any], last_modified: int) -> dbc.Card:
        """Create a status card for uploaded file"""
        
        # Format timestamp
        upload_time = datetime.fromtimestamp(last_modified / 1000).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create metadata display
        metadata_items = []
        for key, value in status_info['metadata'].items():
            if isinstance(value, (int, float, str)):
                metadata_items.append(html.Li(f"{key}: {value}"))
            elif isinstance(value, list) and len(value) < 10:
                metadata_items.append(html.Li(f"{key}: {', '.join(map(str, value))}"))
            elif isinstance(value, dict):
                metadata_items.append(html.Li(f"{key}: {len(value)} items"))
        
        # Determine card color based on data type
        if status_info['is_attack_data']:
            card_color = "danger"
            data_type_badge = dbc.Badge("Attack Data", color="danger", className="me-2")
        elif status_info['is_normal_data']:
            card_color = "success"
            data_type_badge = dbc.Badge("Normal Data", color="success", className="me-2")
        else:
            card_color = "warning"
            data_type_badge = dbc.Badge("Unknown", color="warning", className="me-2")
        
        status_card = dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Span(status_info['status_icon'], className="me-2"),
                    html.Strong(status_info['filename']),
                    html.Small(f" • {upload_time}", className="text-muted ms-2")
                ])
            ]),
            dbc.CardBody([
                html.Div([
                    data_type_badge,
                    dbc.Badge(f"Format: {status_info['format_type']}", color="info", className="me-2"),
                    dbc.Badge(f"Confidence: {status_info['confidence']:.1%}", color="secondary")
                ], className="mb-3"),
                
                html.H6(f"Category: {status_info['data_category'].title()}", className="mb-2"),
                
                html.Details([
                    html.Summary("📊 Metadata", className="mb-2"),
                    html.Ul(metadata_items) if metadata_items else html.P("No metadata available", className="text-muted")
                ])
            ])
        ], color=card_color, outline=True, className="mb-3")
        
        return status_card
    
    def create_preview_component(self, data: Any, status_info: Dict[str, Any], filename: str) -> dbc.Card:
        """Create a preview component for the uploaded data"""
        
        preview_content = []
        
        try:
            # Handle different data types
            if isinstance(data, dict):
                if status_info['format_type'].startswith('json_stix'):
                    preview_content = self.create_stix_preview(data)
                elif status_info['format_type'].startswith('json_mitre'):
                    preview_content = self.create_mitre_preview(data)
                elif status_info['format_type'].startswith('xml_'):
                    preview_content = self.create_xml_preview(data)
                elif status_info['format_type'].startswith('yaml_'):
                    preview_content = self.create_yaml_preview(data)
                elif status_info['format_type'].startswith('npz_'):
                    preview_content = self.create_npz_preview(data)
                else:
                    preview_content = self.create_generic_dict_preview(data)
            
            elif isinstance(data, list):
                if status_info['format_type'].startswith('csv_'):
                    preview_content = self.create_csv_preview(data)
                else:
                    preview_content = self.create_generic_list_preview(data)
            
            elif isinstance(data, np.ndarray):
                preview_content = self.create_numpy_preview(data, status_info)
            
            else:
                preview_content = [html.P(f"Preview not available for {type(data).__name__}", className="text-muted")]
        
        except Exception as e:
            preview_content = [html.P(f"Error creating preview: {e}", className="text-danger")]
        
        preview_card = dbc.Card([
            dbc.CardHeader([
                html.H5(f"📋 Data Preview: {filename}", className="mb-0")
            ]),
            dbc.CardBody(preview_content)
        ], className="mb-3")
        
        return preview_card
    
    def create_stix_preview(self, data: dict) -> List:
        """Create preview for STIX 2.1 data"""
        objects = data.get('objects', [])
        
        if not objects:
            return [html.P("No STIX objects found", className="text-muted")]
        
        # Create summary statistics
        object_types = {}
        for obj in objects:
            obj_type = obj.get('type', 'unknown')
            object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Create pie chart for object types
        fig = px.pie(
            values=list(object_types.values()),
            names=list(object_types.keys()),
            title="STIX Object Types Distribution"
        )
        
        return [
            html.Div([
                html.H6("📊 STIX Bundle Summary"),
                html.P(f"Total Objects: {len(objects)}"),
                html.P(f"STIX Version: {data.get('spec_version', 'Unknown')}"),
                html.P(f"Bundle ID: {data.get('id', 'Unknown')}")
            ]),
            dcc.Graph(figure=fig, style={'height': '300px'}),
            html.Details([
                html.Summary("🔍 First 3 Objects"),
                html.Pre(json.dumps(objects[:3], indent=2), 
                        style={'max-height': '200px', 'overflow-y': 'scroll'})
            ])
        ]
    
    def create_mitre_preview(self, data: dict) -> List:
        """Create preview for MITRE ATT&CK data"""
        return [
            html.Div([
                html.H6("🎯 MITRE ATT&CK Data"),
                html.P(f"Data keys: {len(data.keys())}"),
                html.P("Sample techniques found in data:")
            ]),
            html.Pre(json.dumps(data, indent=2)[:1000] + "..." if len(str(data)) > 1000 else json.dumps(data, indent=2),
                    style={'max-height': '300px', 'overflow-y': 'scroll'})
        ]
    
    def create_csv_preview(self, data: list) -> List:
        """Create preview for CSV data"""
        if not data:
            return [html.P("No data found", className="text-muted")]
        
        # Create DataFrame for display
        df = pd.DataFrame(data[:100])  # Show first 100 rows
        
        return [
            html.Div([
                html.H6("📊 CSV Data Summary"),
                html.P(f"Total rows: {len(data)}"),
                html.P(f"Columns: {len(df.columns)}"),
                html.P(f"Column names: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
            ]),
            html.Div([
                html.H6("🔍 Data Preview (First 10 rows)"),
                dbc.Table.from_dataframe(df.head(10), striped=True, bordered=True, hover=True, size="sm")
            ])
        ]
    
    def create_numpy_preview(self, data: np.ndarray, status_info: Dict[str, Any]) -> List:
        """Create preview for NumPy data"""
        # Create visualization based on data shape
        preview_components = [
            html.Div([
                html.H6("🔢 NumPy Array Summary"),
                html.P(f"Shape: {data.shape}"),
                html.P(f"Data type: {data.dtype}"),
                html.P(f"Size: {data.nbytes / 1024 / 1024:.2f} MB"),
                html.P(f"Min value: {np.min(data):.4f}"),
                html.P(f"Max value: {np.max(data):.4f}"),
                html.P(f"Mean: {np.mean(data):.4f}")
            ])
        ]
        
        # Add visualization if appropriate
        if len(data.shape) == 2 and data.shape[0] <= 50 and data.shape[1] <= 50:
            # Small 2D array - show as heatmap
            fig = px.imshow(data, title="Data Heatmap")
            preview_components.append(dcc.Graph(figure=fig, style={'height': '400px'}))
        
        elif len(data.shape) == 1 and len(data) <= 1000:
            # 1D array - show as line plot
            fig = px.line(y=data, title="Data Values")
            preview_components.append(dcc.Graph(figure=fig, style={'height': '300px'}))
        
        return preview_components
    
    def create_xml_preview(self, data: dict) -> List:
        """Create preview for XML data"""
        return [
            html.Div([
                html.H6("📄 XML Data Structure"),
                html.P(f"Root elements: {len(data.keys())}"),
                html.P(f"Keys: {', '.join(list(data.keys())[:10])}{'...' if len(data.keys()) > 10 else ''}")
            ]),
            html.Pre(json.dumps(data, indent=2)[:1000] + "..." if len(str(data)) > 1000 else json.dumps(data, indent=2),
                    style={'max-height': '300px', 'overflow-y': 'scroll'})
        ]
    
    def create_yaml_preview(self, data: dict) -> List:
        """Create preview for YAML data"""
        return [
            html.Div([
                html.H6("📋 YAML Configuration"),
                html.P(f"Configuration sections: {len(data.keys())}"),
                html.P(f"Keys: {', '.join(list(data.keys())[:10])}{'...' if len(data.keys()) > 10 else ''}")
            ]),
            html.Pre(json.dumps(data, indent=2)[:1000] + "..." if len(str(data)) > 1000 else json.dumps(data, indent=2),
                    style={'max-height': '300px', 'overflow-y': 'scroll'})
        ]
    
    def create_npz_preview(self, data: dict) -> List:
        """Create preview for NPZ data"""
        array_info = []
        for key, array in data.items():
            if isinstance(array, np.ndarray):
                array_info.append(html.Li(f"{key}: {array.shape} {array.dtype}"))
        
        return [
            html.Div([
                html.H6("📦 NPZ Multi-Array Data"),
                html.P(f"Number of arrays: {len(data)}"),
                html.P("Array information:"),
                html.Ul(array_info)
            ])
        ]
    
    def create_generic_dict_preview(self, data: dict) -> List:
        """Create preview for generic dictionary data"""
        return [
            html.Div([
                html.H6("📊 Dictionary Data"),
                html.P(f"Keys: {len(data.keys())}"),
                html.P(f"Key names: {', '.join(list(data.keys())[:10])}{'...' if len(data.keys()) > 10 else ''}")
            ]),
            html.Pre(json.dumps(data, indent=2)[:1000] + "..." if len(str(data)) > 1000 else json.dumps(data, indent=2),
                    style={'max-height': '300px', 'overflow-y': 'scroll'})
        ]
    
    def create_generic_list_preview(self, data: list) -> List:
        """Create preview for generic list data"""
        return [
            html.Div([
                html.H6("📋 List Data"),
                html.P(f"Items: {len(data)}"),
                html.P(f"First item type: {type(data[0]).__name__}" if data else "Empty list")
            ]),
            html.Pre(json.dumps(data[:5], indent=2) + "..." if len(data) > 5 else json.dumps(data, indent=2),
                    style={'max-height': '300px', 'overflow-y': 'scroll'})
        ]
    
    def create_error_card(self, filename: str, error_msg: str) -> dbc.Card:
        """Create error card for failed uploads"""
        return dbc.Card([
            dbc.CardHeader([
                html.Span("❌", className="me-2"),
                html.Strong(filename),
                html.Small(" • Upload Failed", className="text-danger ms-2")
            ]),
            dbc.CardBody([
                html.P("Failed to process file", className="text-danger"),
                html.Details([
                    html.Summary("Error Details"),
                    html.Pre(error_msg, className="text-danger")
                ])
            ])
        ], color="danger", outline=True, className="mb-3")

# Usage example
def integrate_enhanced_upload(app):
    """
    Integrate the enhanced upload component into an existing Dash app
    """
    upload_component = EnhancedUploadComponent()
    upload_component.create_upload_callbacks(app)
    return upload_component.create_upload_component()

if __name__ == "__main__":
    # Test the component
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    upload_component = EnhancedUploadComponent()
    
    app.layout = dbc.Container([
        html.H1("Enhanced Upload System Test"),
        upload_component.create_upload_component()
    ])
    
    upload_component.create_upload_callbacks(app)
    
    app.run_server(debug=True, port=8050) 