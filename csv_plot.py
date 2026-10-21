# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import dash
# from dash import dcc, html, Input, Output, State
# import numpy as np
# import os

# # Initialize the Dash app
# app = dash.Dash(__name__, title="Interactive Data Visualization")

# # Define the layout
# app.layout = html.Div([
#     html.H1("Interactive Data Visualization Dashboard"),
    
#     # File upload component
#     html.Div([
#         html.H3("Upload your CSV file"),
#         dcc.Upload(
#             id='upload-data',
#             children=html.Div([
#                 'Drag and Drop or ',
#                 html.A('Select a CSV File')
#             ]),
#             style={
#                 'width': '100%',
#                 'height': '60px',
#                 'lineHeight': '60px',
#                 'borderWidth': '1px',
#                 'borderStyle': 'dashed',
#                 'borderRadius': '5px',
#                 'textAlign': 'center',
#                 'margin': '10px'
#             },
#             multiple=False
#         ),
#     ]),
    
#     # Display the uploaded data
#     html.Div(id='output-data-upload'),
    
#     # Plot configuration
#     html.Div([
#         html.H3("Configure Your Plot"),
        
#         # Plot type selection
#         html.Div([
#             html.Label("Plot Type:"),
#             dcc.Dropdown(
#                 id='plot-type',
#                 options=[
#                     {'label': '2D Scatter Plot', 'value': 'scatter'},
#                     {'label': '3D Scatter Plot', 'value': 'scatter3d'},
#                     {'label': 'Line Plot', 'value': 'line'},
#                     {'label': 'Bar Chart', 'value': 'bar'},
#                     {'label': 'Histogram', 'value': 'histogram'},
#                     {'label': 'Box Plot', 'value': 'box'},
#                     {'label': 'Heat Map', 'value': 'heatmap'},
#                     {'label': 'Pie Chart', 'value': 'pie'},
#                     {'label': 'Violin Plot', 'value': 'violin'},
#                     {'label': 'Contour Plot', 'value': 'contour'}
#                 ],
#                 value='scatter'
#             ),
#         ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
#         # Variable selection (will be populated after file upload)
#         html.Div([
#             html.Div([
#                 html.Label("X-Axis Variable:"),
#                 dcc.Dropdown(id='x-variable'),
#             ], style={'marginBottom': '10px'}),
            
#             html.Div([
#                 html.Label("Y-Axis Variables (select multiple):"),
#                 dcc.Dropdown(id='y-variables', multi=True),
#             ], style={'marginBottom': '10px'}),
            
#             html.Div([
#                 html.Label("Z-Axis Variable (for 3D plots):"),
#                 dcc.Dropdown(id='z-variable'),
#             ], style={'marginBottom': '10px'}),
            
#             html.Div([
#                 html.Label("Color Variable (optional, for single Y variable):"),
#                 dcc.Dropdown(id='color-variable'),
#             ], style={'marginBottom': '10px'}),
            
#             html.Div([
#                 html.Label("Size Variable (optional):"),
#                 dcc.Dropdown(id='size-variable'),
#             ], style={'marginBottom': '10px'}),
            
#             # New: Hover info selection
#             html.Div([
#                 html.Label("Additional Hover Info (select multiple):"),
#                 dcc.Dropdown(id='hover-variables', multi=True, placeholder="Select columns to show on hover"),
#             ]),
#         ], style={'width': '48%', 'display': 'inline-block'}),
        
#         html.Button('Generate Plot', id='generate-button', n_clicks=0, 
#                    style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px',
#                          'margin': '20px 0px', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
#     ], id='plot-controls', style={'display': 'none'}),
    
#     # Plot display
#     html.Div([
#         dcc.Graph(id='plot-output', style={'height': '800px'})
#     ], id='plot-container', style={'display': 'none'}),
    
#     # Data table display
#     html.Div(id='data-table-container', style={'margin': '20px 0px'}),
    
#     # Download options
#     html.Div([
#         html.Button('Download Plot as PNG', id='download-button', 
#                    style={'backgroundColor': '#008CBA', 'color': 'white', 'padding': '10px 20px',
#                          'margin': '20px 10px', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
#     ], id='download-container', style={'display': 'none'}),
    
#     # Store the dataframe in a hidden div
#     dcc.Store(id='stored-data'),
# ])

# # Function to parse uploaded CSV
# def parse_contents(contents, filename):
#     import base64
#     import io
    
#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
    
#     try:
#         if 'csv' in filename:
#             # Assume that the user uploaded a CSV file
#             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#         elif 'xls' in filename:
#             # Assume that the user uploaded an excel file
#             df = pd.read_excel(io.BytesIO(decoded))
#         else:
#             return None, "Unsupported file type. Please upload a CSV or Excel file."
            
#         return df, ""
#     except Exception as e:
#         return None, f'Error processing file: {str(e)}'

# # Callback to update the data store and show data preview
# @app.callback(
#     [Output('stored-data', 'data'),
#      Output('output-data-upload', 'children'),
#      Output('plot-controls', 'style'),
#      Output('x-variable', 'options'),
#      Output('y-variables', 'options'),
#      Output('z-variable', 'options'),
#      Output('color-variable', 'options'),
#      Output('size-variable', 'options'),
#      Output('hover-variables', 'options')],  # New output for hover variables
#     [Input('upload-data', 'contents')],
#     [State('upload-data', 'filename')]
# )
# def update_output(contents, filename):
#     if contents is None:
#         return None, [], {'display': 'none'}, [], [], [], [], [], []  # Added empty list for hover options
    
#     df, error_message = parse_contents(contents, filename)
    
#     if df is None:
#         return None, html.Div([
#             html.H5(error_message),
#         ]), {'display': 'none'}, [], [], [], [], [], []  # Added empty list for hover options
    
#     # Create dropdown options from dataframe columns
#     columns = df.columns
#     dropdown_options = [{'label': col, 'value': col} for col in columns]
    
#     # Create a data preview table
#     data_preview = html.Div([
#         html.H5(f'Data Preview: {filename}'),
#         html.P(f'Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.'),
        
#         # Display summary statistics
#         html.Details([
#             html.Summary('Dataset Summary Statistics'),
#             html.Pre(df.describe().to_string(), style={'whiteSpace': 'pre-wrap'})
#         ]),
        
#         # Display a preview of the data
#         html.Div([
#             html.H6('First 5 rows:'),
#             html.Table(
#                 # Header
#                 [html.Tr([html.Th(col) for col in df.columns])] +
                
#                 # Body
#                 [html.Tr([
#                     html.Td(df.iloc[i][col]) for col in df.columns
#                 ]) for i in range(min(5, len(df)))]
#             )
#         ]),
#     ])
    
#     return df.to_dict('records'), data_preview, {'display': 'block'}, dropdown_options, dropdown_options, dropdown_options, dropdown_options, dropdown_options, dropdown_options

# # Callback to generate plot
# @app.callback(
#     [Output('plot-output', 'figure'),
#      Output('plot-container', 'style'),
#      Output('download-container', 'style')],
#     [Input('generate-button', 'n_clicks')],
#     [State('stored-data', 'data'),
#      State('plot-type', 'value'),
#      State('x-variable', 'value'),
#      State('y-variables', 'value'),
#      State('z-variable', 'value'),
#      State('color-variable', 'value'),
#      State('size-variable', 'value'),
#      State('hover-variables', 'value')]  # New state for hover variables
# )
# def update_graph(n_clicks, data, plot_type, x_var, y_vars, z_var, color_var, size_var, hover_vars):
#     if n_clicks == 0 or data is None or x_var is None:
#         # Return empty figure if button not clicked or data/x-var not selected
#         return {}, {'display': 'none'}, {'display': 'none'}
    
#     # Convert stored data back to dataframe
#     df = pd.DataFrame(data)
    
#     # Check if y_vars is a list and not empty
#     if not y_vars or not isinstance(y_vars, list):
#         y_vars = [y_vars] if y_vars else []
    
#     # Ensure hover_vars is a list
#     if not hover_vars:
#         hover_vars = []
#     elif not isinstance(hover_vars, list):
#         hover_vars = [hover_vars]
    
#     # Create figure based on plot type
#     fig = None
    
#     # Define a function to create hover text with additional variables
#     def create_hover_text(row, base_vars, hover_vars):
#         # Start with the base variables (the ones being plotted)
#         hover_text = "<br>".join([f"{var}: {row[var]}" for var in base_vars if var in row])
        
#         # Add the additional hover variables
#         if hover_vars:
#             hover_text += "<br>" + "<br>".join([f"{var}: {row[var]}" for var in hover_vars if var in row])
        
#         return hover_text
    
#     # For multiple Y variables, we'll use different approach based on plot type
#     if len(y_vars) > 1:
#         # For most plot types with multiple Y variables, we'll create a figure with multiple traces
        
#         if plot_type in ['scatter', 'line', 'bar']:
#             fig = go.Figure()
            
#             # Add a trace for each Y variable
#             for i, y_var in enumerate(y_vars):
#                 # Base hover variables
#                 base_vars = [x_var, y_var]
                
#                 # Create hover text array if hover_vars are specified
#                 if hover_vars:
#                     hover_texts = [create_hover_text(row, base_vars, hover_vars) for _, row in df.iterrows()]
#                 else:
#                     hover_texts = None
                
#                 if plot_type == 'scatter':
#                     fig.add_trace(go.Scatter(
#                         x=df[x_var], 
#                         y=df[y_var],
#                         mode='markers',
#                         name=y_var,
#                         marker=dict(
#                             size=df[size_var] if size_var else 8
#                         ),
#                         hovertext=hover_texts,
#                         hoverinfo='text' if hover_texts else 'x+y+name'
#                     ))
#                 elif plot_type == 'line':
#                     fig.add_trace(go.Scatter(
#                         x=df[x_var], 
#                         y=df[y_var],
#                         mode='lines+markers',
#                         name=y_var,
#                         hovertext=hover_texts,
#                         hoverinfo='text' if hover_texts else 'x+y+name'
#                     ))
#                 elif plot_type == 'bar':
#                     fig.add_trace(go.Bar(
#                         x=df[x_var], 
#                         y=df[y_var],
#                         name=y_var,
#                         hovertext=hover_texts,
#                         hoverinfo='text' if hover_texts else 'x+y+name'
#                     ))
            
#             # Update layout
#             fig.update_layout(
#                 title=f'Multiple {plot_type.capitalize()} Plot',
#                 xaxis_title=x_var,
#                 yaxis_title='Values',
#                 legend_title='Variables',
#                 template='plotly_white',
#                 height=700,
#                 hovermode='closest'
#             )
            
#         elif plot_type == 'box':
#             fig = go.Figure()
            
#             for y_var in y_vars:
#                 # For box plots, hover info is different
#                 hover_info = []
#                 for i, row in df.iterrows():
#                     base_info = f"{y_var}: {row[y_var]}"
#                     additional_info = "<br>".join([f"{var}: {row[var]}" for var in hover_vars if var in row]) if hover_vars else ""
#                     hover_info.append(f"{base_info}<br>{additional_info}" if additional_info else base_info)
                
#                 fig.add_trace(go.Box(
#                     y=df[y_var],
#                     name=y_var,
#                     hovertext=hover_info if hover_vars else None,
#                     hoverinfo='text' if hover_vars else 'y+name'
#                 ))
            
#             fig.update_layout(
#                 title='Multiple Box Plots',
#                 yaxis_title='Values',
#                 template='plotly_white',
#                 height=700
#             )
            
#         elif plot_type == 'histogram':
#             fig = go.Figure()
            
#             for y_var in y_vars:
#                 fig.add_trace(go.Histogram(
#                     x=df[y_var],
#                     name=y_var,
#                     opacity=0.7,
#                     # Histograms don't support custom hover text per point
#                     hovertemplate=f"{y_var}: %{{x}}<br>Count: %{{y}}<extra></extra>"
#                 ))
            
#             # Overlay histograms
#             fig.update_layout(
#                 title='Multiple Histograms',
#                 xaxis_title='Values',
#                 yaxis_title='Count',
#                 barmode='overlay',
#                 template='plotly_white',
#                 height=700
#             )
            
#         elif plot_type == 'violin':
#             fig = go.Figure()
            
#             for y_var in y_vars:
#                 # For violin plots, hover info is mainly statistical
#                 fig.add_trace(go.Violin(
#                     y=df[y_var],
#                     name=y_var,
#                     box_visible=True,
#                     meanline_visible=True,
#                     hoverinfo='y+name'  # Violin plots don't support custom hovertext for each point
#                 ))
            
#             fig.update_layout(
#                 title='Multiple Violin Plots',
#                 yaxis_title='Values',
#                 template='plotly_white',
#                 height=700
#             )
            
#         elif plot_type == 'scatter3d':
#             # For 3D scatter with multiple Y variables, we'll use the first as Y and color by the others
#             if not z_var:
#                 return {'data': [], 'layout': {'title': 'Error: Z-axis variable required for 3D plot'}}, {'display': 'block'}, {'display': 'block'}
            
#             fig = go.Figure()
            
#             # Use the first Y variable for the y-axis
#             primary_y = y_vars[0]
            
#             # Base hover variables
#             base_vars = [x_var, primary_y, z_var]
            
#             # Create hover text array if hover_vars are specified
#             if hover_vars:
#                 hover_texts = [create_hover_text(row, base_vars, hover_vars) for _, row in df.iterrows()]
#             else:
#                 hover_texts = None
            
#             # Add a trace for each additional Y variable as color
#             for i, secondary_y in enumerate(y_vars[1:], 1):
#                 fig.add_trace(go.Scatter3d(
#                     x=df[x_var],
#                     y=df[primary_y],
#                     z=df[z_var],
#                     mode='markers',
#                     marker=dict(
#                         size=8,
#                         color=df[secondary_y],
#                         colorscale='Viridis',
#                         colorbar=dict(
#                             title=secondary_y,
#                             x=0.9 + (i-1)*0.1  # Offset colorbar position for each trace
#                         ),
#                         opacity=0.8
#                     ),
#                     name=f'Color: {secondary_y}',
#                     hovertext=hover_texts,
#                     hoverinfo='text' if hover_texts else 'x+y+z+name'
#                 ))
            
#             fig.update_layout(
#                 title='3D Scatter Plot with Multiple Variables',
#                 scene=dict(
#                     xaxis_title=x_var,
#                     yaxis_title=primary_y,
#                     zaxis_title=z_var
#                 ),
#                 template='plotly_white',
#                 height=700
#             )
            
#         else:
#             # For other plot types not supporting multiple Y variables directly
#             return {'data': [], 'layout': {'title': 'Error: Multiple Y variables not supported for this plot type'}}, {'display': 'block'}, {'display': 'block'}
    
#     else:
#         # Single Y variable case
#         # Get single Y variable if available
#         y_var = y_vars[0] if y_vars and len(y_vars) > 0 else None
        
#         # For Plotly Express plots, we can use the hover_data parameter
#         hover_data = hover_vars if hover_vars else None
        
#         # Common keyword arguments for most plots
#         kwargs = {
#             'title': f'{plot_type.capitalize()} Plot',
#             'labels': {x_var: x_var, y_var: y_var} if y_var else {x_var: x_var},
#             'hover_data': hover_data  # Add hover_data parameter
#         }
        
#         # Add color variable if provided
#         if color_var:
#             kwargs['color'] = df[color_var]
#             kwargs['labels'][color_var] = color_var
        
#         # Add size variable if provided and applicable
#         if size_var and plot_type in ['scatter', 'scatter3d']:
#             kwargs['size'] = df[size_var]
#             kwargs['labels'][size_var] = size_var
        
#         # Create appropriate plot based on selection
#         if plot_type == 'scatter':
#             if y_var:
#                 fig = px.scatter(df, x=x_var, y=y_var, **kwargs)
#             else:
#                 fig = px.scatter(df, x=x_var, **kwargs)
        
#         elif plot_type == 'scatter3d':
#             if not z_var:
#                 # Fallback if z variable is not provided
#                 return {'data': [], 'layout': {'title': 'Error: Z-axis variable required for 3D plot'}}, {'display': 'block'}, {'display': 'block'}
            
#             kwargs['labels'][z_var] = z_var
#             fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var, **kwargs)
        
#         elif plot_type == 'line':
#             if y_var:
#                 fig = px.line(df, x=x_var, y=y_var, **kwargs)
#             else:
#                 fig = px.line(df, x=df.index, y=x_var, **kwargs)
        
#         elif plot_type == 'bar':
#             if y_var:
#                 fig = px.bar(df, x=x_var, y=y_var, **kwargs)
#             else:
#                 fig = px.bar(df, x=df.index, y=x_var, **kwargs)
        
#         elif plot_type == 'histogram':
#             fig = px.histogram(df, x=x_var, y=y_var if y_var else None, **kwargs)
        
#         elif plot_type == 'box':
#             if y_var:
#                 fig = px.box(df, x=x_var, y=y_var, **kwargs)
#             else:
#                 fig = px.box(df, y=x_var, **kwargs)
        
#         elif plot_type == 'heatmap':
#             if not y_var:
#                 return {'data': [], 'layout': {'title': 'Error: Y-axis variable required for heatmap'}}, {'display': 'block'}, {'display': 'block'}
            
#             # For heatmap, hover_data doesn't apply in the same way
#             if 'hover_data' in kwargs:
#                 del kwargs['hover_data']
            
#             # For heatmap, we need to pivot the data
#             pivot_data = df.pivot_table(
#                 index=y_var, 
#                 columns=x_var, 
#                 values=z_var if z_var else color_var if color_var else size_var,
#                 aggfunc='mean'
#             )
            
#             fig = px.imshow(
#                 pivot_data, 
#                 labels=dict(color=z_var if z_var else color_var if color_var else size_var),
#                 title=f'Heatmap of {z_var if z_var else color_var if color_var else size_var}'
#             )
        
#         elif plot_type == 'pie':
#             fig = px.pie(df, names=x_var, values=y_var if y_var else None, **kwargs)
        
#         elif plot_type == 'violin':
#             if y_var:
#                 fig = px.violin(df, x=x_var, y=y_var, **kwargs)
#             else:
#                 fig = px.violin(df, y=x_var, **kwargs)
        
#         elif plot_type == 'contour':
#             if not y_var or not z_var:
#                 return {'data': [], 'layout': {'title': 'Error: X, Y, and Z variables required for contour plot'}}, {'display': 'block'}, {'display': 'block'}
            
#             # For contour, hover_data doesn't apply in the same way
#             if 'hover_data' in kwargs:
#                 del kwargs['hover_data']
            
#             # For contour, we need to pivot the data
#             pivot_data = df.pivot_table(
#                 index=y_var, 
#                 columns=x_var, 
#                 values=z_var,
#                 aggfunc='mean'
#             )
            
#             fig = px.imshow(
#                 pivot_data,
#                 labels=dict(color=z_var),
#                 title=f'Contour Plot of {z_var}'
#             )
    
#     # Update layout for better appearance
#     if fig:
#         fig.update_layout(
#             template='plotly_white',
#             height=700,
#             legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
#             margin=dict(l=40, r=40, t=60, b=40),
#             hovermode='closest'  # Ensure the closest point is always shown in hover
#         )
        
#         # Customize hover template if needed
#         if hover_vars and hasattr(fig, 'data'):
#             for trace in fig.data:
#                 if hasattr(trace, 'hovertemplate') and not trace.hovertemplate:
#                     # This is a simple way to ensure basic hover info is shown
#                     # along with any custom hover data
#                     trace.hovertemplate = '%{hovertext}<extra></extra>'
    
#     return fig, {'display': 'block'}, {'display': 'block'}

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import numpy as np
import os

# Initialize the Dash app
app = dash.Dash(__name__, title="Interactive Data Visualization")

# Define the layout
app.layout = html.Div([
    html.H1("Interactive Data Visualization Dashboard"),
    
    # File upload component
    html.Div([
        html.H3("Upload your CSV file"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a CSV File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False
        ),
    ]),
    
    # Display the uploaded data
    html.Div(id='output-data-upload'),
    
    # Plot configuration
    html.Div([
        html.H3("Configure Your Plot"),
        
        # Plot type selection
        html.Div([
            html.Label("Plot Type:"),
            dcc.Dropdown(
                id='plot-type',
                options=[
                    {'label': '2D Scatter Plot', 'value': 'scatter'},
                    {'label': '3D Scatter Plot', 'value': 'scatter3d'},
                    {'label': 'Line Plot', 'value': 'line'},
                    {'label': 'Bar Chart', 'value': 'bar'},
                    {'label': 'Histogram', 'value': 'histogram'},
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Heat Map', 'value': 'heatmap'},
                    {'label': 'Pie Chart', 'value': 'pie'},
                    {'label': 'Violin Plot', 'value': 'violin'},
                    {'label': 'Contour Plot', 'value': 'contour'}
                ],
                value='scatter'
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        # Variable selection (will be populated after file upload)
        html.Div([
            html.Div([
                html.Label("X-Axis Variable:"),
                dcc.Dropdown(id='x-variable'),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("Y-Axis Variables (select multiple):"),
                dcc.Dropdown(id='y-variables', multi=True),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("Z-Axis Variable (for 3D plots):"),
                dcc.Dropdown(id='z-variable'),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("Color Variable (optional):"),
                dcc.Dropdown(id='color-variable'),
                html.P("Note: For multiple Y variables, color will apply to all traces", 
                       style={'fontSize': '12px', 'color': 'gray', 'margin': '5px 0'}),
            ], style={'marginBottom': '10px'}),
            
            html.Div([
                html.Label("Size Variable (optional):"),
                dcc.Dropdown(id='size-variable'),
            ], style={'marginBottom': '10px'}),
            
            # New: Hover info selection
            html.Div([
                html.Label("Additional Hover Info (select multiple):"),
                dcc.Dropdown(id='hover-variables', multi=True, placeholder="Select columns to show on hover"),
            ]),
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Button('Generate Plot', id='generate-button', n_clicks=0, 
                   style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px',
                         'margin': '20px 0px', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
    ], id='plot-controls', style={'display': 'none'}),
    
    # Plot display
    html.Div([
        dcc.Graph(id='plot-output', style={'height': '800px'})
    ], id='plot-container', style={'display': 'none'}),
    
    # Data table display
    html.Div(id='data-table-container', style={'margin': '20px 0px'}),
    
    # Download options
    html.Div([
        html.Button('Download Plot as PNG', id='download-button', 
                   style={'backgroundColor': '#008CBA', 'color': 'white', 'padding': '10px 20px',
                         'margin': '20px 10px', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px'}),
    ], id='download-container', style={'display': 'none'}),
    
    # Store the dataframe in a hidden div
    dcc.Store(id='stored-data'),
])

# Function to parse uploaded CSV
def parse_contents(contents, filename):
    import base64
    import io
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, "Unsupported file type. Please upload a CSV or Excel file."
            
        return df, ""
    except Exception as e:
        return None, f'Error processing file: {str(e)}'

# Callback to update the data store and show data preview
@app.callback(
    [Output('stored-data', 'data'),
     Output('output-data-upload', 'children'),
     Output('plot-controls', 'style'),
     Output('x-variable', 'options'),
     Output('y-variables', 'options'),
     Output('z-variable', 'options'),
     Output('color-variable', 'options'),
     Output('size-variable', 'options'),
     Output('hover-variables', 'options')],  # New output for hover variables
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is None:
        return None, [], {'display': 'none'}, [], [], [], [], [], []  # Added empty list for hover options
    
    df, error_message = parse_contents(contents, filename)
    
    if df is None:
        return None, html.Div([
            html.H5(error_message),
        ]), {'display': 'none'}, [], [], [], [], [], []  # Added empty list for hover options
    
    # Create dropdown options from dataframe columns
    columns = df.columns
    dropdown_options = [{'label': col, 'value': col} for col in columns]
    
    # Create a data preview table
    data_preview = html.Div([
        html.H5(f'Data Preview: {filename}'),
        html.P(f'Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.'),
        
        # Display summary statistics
        html.Details([
            html.Summary('Dataset Summary Statistics'),
            html.Pre(df.describe().to_string(), style={'whiteSpace': 'pre-wrap'})
        ]),
        
        # Display a preview of the data
        html.Div([
            html.H6('First 5 rows:'),
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in df.columns])] +
                
                # Body
                [html.Tr([
                    html.Td(df.iloc[i][col]) for col in df.columns
                ]) for i in range(min(5, len(df)))]
            )
        ]),
    ])
    
    return df.to_dict('records'), data_preview, {'display': 'block'}, dropdown_options, dropdown_options, dropdown_options, dropdown_options, dropdown_options, dropdown_options

# Function to create color mapping for categorical variables
def get_color_mapping(df, color_var):
    """Create a consistent color mapping for categorical variables"""
    if color_var is None or color_var not in df.columns:
        return None, None
    
    unique_values = df[color_var].unique()
    # Use plotly's default color sequence
    colors = px.colors.qualitative.Plotly
    color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}
    return unique_values, color_map

# Callback to generate plot
@app.callback(
    [Output('plot-output', 'figure'),
     Output('plot-container', 'style'),
     Output('download-container', 'style')],
    [Input('generate-button', 'n_clicks')],
    [State('stored-data', 'data'),
     State('plot-type', 'value'),
     State('x-variable', 'value'),
     State('y-variables', 'value'),
     State('z-variable', 'value'),
     State('color-variable', 'value'),
     State('size-variable', 'value'),
     State('hover-variables', 'value')]  # New state for hover variables
)
def update_graph(n_clicks, data, plot_type, x_var, y_vars, z_var, color_var, size_var, hover_vars):
    if n_clicks == 0 or data is None or x_var is None:
        # Return empty figure if button not clicked or data/x-var not selected
        return {}, {'display': 'none'}, {'display': 'none'}
    
    # Convert stored data back to dataframe
    df = pd.DataFrame(data)
    
    # Check if y_vars is a list and not empty
    if not y_vars or not isinstance(y_vars, list):
        y_vars = [y_vars] if y_vars else []
    
    # Ensure hover_vars is a list
    if not hover_vars:
        hover_vars = []
    elif not isinstance(hover_vars, list):
        hover_vars = [hover_vars]
    
    # Get color mapping for categorical variables
    unique_colors, color_map = get_color_mapping(df, color_var)
    
    # Create figure based on plot type
    fig = None
    
    # Define a function to create hover text with additional variables
    def create_hover_text(row, base_vars, hover_vars, include_color=True):
        # Start with the base variables (the ones being plotted)
        hover_text = "<br>".join([f"{var}: {row[var]}" for var in base_vars if var in row])
        
        # Add color variable if it exists and is not already in base_vars
        if include_color and color_var and color_var not in base_vars and color_var in row:
            hover_text += f"<br>{color_var}: {row[color_var]}"
        
        # Add the additional hover variables
        if hover_vars:
            hover_text += "<br>" + "<br>".join([f"{var}: {row[var]}" for var in hover_vars if var in row and var not in base_vars and var != color_var])
        
        return hover_text
    
    # For multiple Y variables, we'll use different approach based on plot type
    if len(y_vars) > 1:
        # For most plot types with multiple Y variables, we'll create a figure with multiple traces
        
        if plot_type in ['scatter', 'line', 'bar']:
            fig = go.Figure()
            
            # Add a trace for each Y variable
            for i, y_var in enumerate(y_vars):
                # Base hover variables
                base_vars = [x_var, y_var]
                
                # Create hover text array if hover_vars are specified
                hover_texts = [create_hover_text(row, base_vars, hover_vars) for _, row in df.iterrows()]
                
                # Handle coloring for multiple Y variables
                if color_var and color_var in df.columns:
                    # Group data by color variable and create separate traces
                    for color_val in unique_colors:
                        mask = df[color_var] == color_val
                        if mask.sum() == 0:  # Skip if no data for this color value
                            continue
                            
                        subset_df = df[mask]
                        subset_hover = [create_hover_text(row, base_vars, hover_vars) for _, row in subset_df.iterrows()]
                        
                        trace_name = f"{y_var} - {color_val}"
                        trace_color = color_map[color_val]
                        
                        if plot_type == 'scatter':
                            fig.add_trace(go.Scatter(
                                x=subset_df[x_var], 
                                y=subset_df[y_var],
                                mode='markers',
                                name=trace_name,
                                marker=dict(
                                    color=trace_color,
                                    size=subset_df[size_var] if size_var else 8
                                ),
                                hovertext=subset_hover,
                                hoverinfo='text',
                                legendgroup=y_var,  # Group by Y variable in legend
                                showlegend=True
                            ))
                        elif plot_type == 'line':
                            fig.add_trace(go.Scatter(
                                x=subset_df[x_var], 
                                y=subset_df[y_var],
                                mode='lines+markers',
                                name=trace_name,
                                line=dict(color=trace_color),
                                marker=dict(color=trace_color),
                                hovertext=subset_hover,
                                hoverinfo='text',
                                legendgroup=y_var,
                                showlegend=True
                            ))
                        elif plot_type == 'bar':
                            fig.add_trace(go.Bar(
                                x=subset_df[x_var], 
                                y=subset_df[y_var],
                                name=trace_name,
                                marker=dict(color=trace_color),
                                hovertext=subset_hover,
                                hoverinfo='text',
                                legendgroup=y_var,
                                showlegend=True
                            ))
                else:
                    # No color variable - just different colors for different Y variables
                    if plot_type == 'scatter':
                        fig.add_trace(go.Scatter(
                            x=df[x_var], 
                            y=df[y_var],
                            mode='markers',
                            name=y_var,
                            marker=dict(
                                size=df[size_var] if size_var else 8
                            ),
                            hovertext=hover_texts,
                            hoverinfo='text'
                        ))
                    elif plot_type == 'line':
                        fig.add_trace(go.Scatter(
                            x=df[x_var], 
                            y=df[y_var],
                            mode='lines+markers',
                            name=y_var,
                            hovertext=hover_texts,
                            hoverinfo='text'
                        ))
                    elif plot_type == 'bar':
                        fig.add_trace(go.Bar(
                            x=df[x_var], 
                            y=df[y_var],
                            name=y_var,
                            hovertext=hover_texts,
                            hoverinfo='text'
                        ))
            
            # Update layout
            fig.update_layout(
                title=f'Multiple {plot_type.capitalize()} Plot',
                xaxis_title=x_var,
                yaxis_title='Values',
                legend_title='Variables',
                template='plotly_white',
                height=700,
                hovermode='closest'
            )
            
        elif plot_type == 'box':
            fig = go.Figure()
            
            for y_var in y_vars:
                if color_var and color_var in df.columns:
                    # Create separate box plots for each color category
                    for color_val in unique_colors:
                        mask = df[color_var] == color_val
                        if mask.sum() == 0:
                            continue
                            
                        subset_df = df[mask]
                        hover_info = []
                        for _, row in subset_df.iterrows():
                            base_info = f"{y_var}: {row[y_var]}"
                            color_info = f"{color_var}: {row[color_var]}"
                            additional_info = "<br>".join([f"{var}: {row[var]}" for var in hover_vars if var in row]) if hover_vars else ""
                            hover_info.append(f"{base_info}<br>{color_info}" + (f"<br>{additional_info}" if additional_info else ""))
                        
                        fig.add_trace(go.Box(
                            y=subset_df[y_var],
                            name=f"{y_var} - {color_val}",
                            marker=dict(color=color_map[color_val]),
                            hovertext=hover_info,
                            hoverinfo='text',
                            legendgroup=y_var,
                            showlegend=True
                        ))
                else:
                    # No color variable
                    hover_info = []
                    for _, row in df.iterrows():
                        base_info = f"{y_var}: {row[y_var]}"
                        additional_info = "<br>".join([f"{var}: {row[var]}" for var in hover_vars if var in row]) if hover_vars else ""
                        hover_info.append(f"{base_info}" + (f"<br>{additional_info}" if additional_info else ""))
                    
                    fig.add_trace(go.Box(
                        y=df[y_var],
                        name=y_var,
                        hovertext=hover_info,
                        hoverinfo='text'
                    ))
            
            fig.update_layout(
                title='Multiple Box Plots',
                yaxis_title='Values',
                template='plotly_white',
                height=700
            )
            
        elif plot_type == 'histogram':
            fig = go.Figure()
            
            for y_var in y_vars:
                if color_var and color_var in df.columns:
                    # Create separate histograms for each color category
                    for color_val in unique_colors:
                        mask = df[color_var] == color_val
                        if mask.sum() == 0:
                            continue
                            
                        subset_df = df[mask]
                        fig.add_trace(go.Histogram(
                            x=subset_df[y_var],
                            name=f"{y_var} - {color_val}",
                            marker=dict(color=color_map[color_val]),
                            opacity=0.7,
                            legendgroup=y_var,
                            showlegend=True,
                            hovertemplate=f"{y_var}: %{{x}}<br>Count: %{{y}}<br>{color_var}: {color_val}<extra></extra>"
                        ))
                else:
                    fig.add_trace(go.Histogram(
                        x=df[y_var],
                        name=y_var,
                        opacity=0.7,
                        hovertemplate=f"{y_var}: %{{x}}<br>Count: %{{y}}<extra></extra>"
                    ))
            
            # Overlay histograms
            fig.update_layout(
                title='Multiple Histograms',
                xaxis_title='Values',
                yaxis_title='Count',
                barmode='overlay',
                template='plotly_white',
                height=700
            )
            
        elif plot_type == 'violin':
            fig = go.Figure()
            
            for y_var in y_vars:
                if color_var and color_var in df.columns:
                    # Create separate violin plots for each color category
                    for color_val in unique_colors:
                        mask = df[color_var] == color_val
                        if mask.sum() == 0:
                            continue
                            
                        subset_df = df[mask]
                        fig.add_trace(go.Violin(
                            y=subset_df[y_var],
                            name=f"{y_var} - {color_val}",
                            marker=dict(color=color_map[color_val]),
                            box_visible=True,
                            meanline_visible=True,
                            legendgroup=y_var,
                            showlegend=True
                        ))
                else:
                    fig.add_trace(go.Violin(
                        y=df[y_var],
                        name=y_var,
                        box_visible=True,
                        meanline_visible=True
                    ))
            
            fig.update_layout(
                title='Multiple Violin Plots',
                yaxis_title='Values',
                template='plotly_white',
                height=700
            )
            
        elif plot_type == 'scatter3d':
            # For 3D scatter with multiple Y variables, we'll use the first as Y and color by the others
            if not z_var:
                return {'data': [], 'layout': {'title': 'Error: Z-axis variable required for 3D plot'}}, {'display': 'block'}, {'display': 'block'}
            
            fig = go.Figure()
            
            # Use the first Y variable for the y-axis
            primary_y = y_vars[0]
            base_vars = [x_var, primary_y, z_var]
            
            if color_var and color_var in df.columns:
                # Color by the color variable
                for color_val in unique_colors:
                    mask = df[color_var] == color_val
                    if mask.sum() == 0:
                        continue
                        
                    subset_df = df[mask]
                    hover_texts = [create_hover_text(row, base_vars, hover_vars) for _, row in subset_df.iterrows()]
                    
                    fig.add_trace(go.Scatter3d(
                        x=subset_df[x_var],
                        y=subset_df[primary_y],
                        z=subset_df[z_var],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_map[color_val]
                        ),
                        name=f"{primary_y} - {color_val}",
                        hovertext=hover_texts,
                        hoverinfo='text'
                    ))
            else:
                # No color variable - just use the first Y variable
                hover_texts = [create_hover_text(row, base_vars, hover_vars, include_color=False) for _, row in df.iterrows()]
                
                fig.add_trace(go.Scatter3d(
                    x=df[x_var],
                    y=df[primary_y],
                    z=df[z_var],
                    mode='markers',
                    marker=dict(size=8),
                    name=primary_y,
                    hovertext=hover_texts,
                    hoverinfo='text'
                ))
            
            fig.update_layout(
                title='3D Scatter Plot',
                scene=dict(
                    xaxis_title=x_var,
                    yaxis_title=primary_y,
                    zaxis_title=z_var
                ),
                template='plotly_white',
                height=700
            )
            
        else:
            # For other plot types not supporting multiple Y variables directly
            return {'data': [], 'layout': {'title': 'Error: Multiple Y variables not supported for this plot type'}}, {'display': 'block'}, {'display': 'block'}
    
    else:
        # Single Y variable case
        # Get single Y variable if available
        y_var = y_vars[0] if y_vars and len(y_vars) > 0 else None
        
        # For Plotly Express plots, we can use the hover_data parameter
        hover_data = hover_vars if hover_vars else None
        
        # Common keyword arguments for most plots
        kwargs = {
            'title': f'{plot_type.capitalize()} Plot',
            'labels': {x_var: x_var, y_var: y_var} if y_var else {x_var: x_var},
            'hover_data': hover_data  # Add hover_data parameter
        }
        
        # Add color variable if provided
        if color_var:
            kwargs['color'] = df[color_var]
            kwargs['labels'][color_var] = color_var
        
        # Add size variable if provided and applicable
        if size_var and plot_type in ['scatter', 'scatter3d']:
            kwargs['size'] = df[size_var]
            kwargs['labels'][size_var] = size_var
        
        # Create appropriate plot based on selection
        if plot_type == 'scatter':
            if y_var:
                fig = px.scatter(df, x=x_var, y=y_var, **kwargs)
            else:
                fig = px.scatter(df, x=x_var, **kwargs)
        
        elif plot_type == 'scatter3d':
            if not z_var:
                # Fallback if z variable is not provided
                return {'data': [], 'layout': {'title': 'Error: Z-axis variable required for 3D plot'}}, {'display': 'block'}, {'display': 'block'}
            
            kwargs['labels'][z_var] = z_var
            fig = px.scatter_3d(df, x=x_var, y=y_var, z=z_var, **kwargs)
        
        elif plot_type == 'line':
            if y_var:
                fig = px.line(df, x=x_var, y=y_var, **kwargs)
            else:
                fig = px.line(df, x=df.index, y=x_var, **kwargs)
        
        elif plot_type == 'bar':
            if y_var:
                fig = px.bar(df, x=x_var, y=y_var, **kwargs)
            else:
                fig = px.bar(df, x=df.index, y=x_var, **kwargs)
        
        elif plot_type == 'histogram':
            fig = px.histogram(df, x=x_var, y=y_var if y_var else None, **kwargs)
        
        elif plot_type == 'box':
            if y_var:
                fig = px.box(df, x=x_var, y=y_var, **kwargs)
            else:
                fig = px.box(df, y=x_var, **kwargs)
        
        elif plot_type == 'heatmap':
            if not y_var:
                return {'data': [], 'layout': {'title': 'Error: Y-axis variable required for heatmap'}}, {'display': 'block'}, {'display': 'block'}
            
            # For heatmap, hover_data doesn't apply in the same way
            if 'hover_data' in kwargs:
                del kwargs['hover_data']
            
            # For heatmap, we need to pivot the data
            pivot_data = df.pivot_table(
                index=y_var, 
                columns=x_var, 
                values=z_var if z_var else color_var if color_var else size_var,
                aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot_data, 
                labels=dict(color=z_var if z_var else color_var if color_var else size_var),
                title=f'Heatmap of {z_var if z_var else color_var if color_var else size_var}'
            )
        
        elif plot_type == 'pie':
            fig = px.pie(df, names=x_var, values=y_var if y_var else None, **kwargs)
        
        elif plot_type == 'violin':
            if y_var:
                fig = px.violin(df, x=x_var, y=y_var, **kwargs)
            else:
                fig = px.violin(df, y=x_var, **kwargs)
        
        elif plot_type == 'contour':
            if not y_var or not z_var:
                return {'data': [], 'layout': {'title': 'Error: X, Y, and Z variables required for contour plot'}}, {'display': 'block'}, {'display': 'block'}
            
            # For contour, hover_data doesn't apply in the same way
            if 'hover_data' in kwargs:
                del kwargs['hover_data']
            
            # For contour, we need to pivot the data
            pivot_data = df.pivot_table(
                index=y_var, 
                columns=x_var, 
                values=z_var,
                aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot_data,
                labels=dict(color=z_var),
                title=f'Contour Plot of {z_var}'
            )
    
    # Update layout for better appearance
    if fig:
        fig.update_layout(
            template='plotly_white',
            height=700,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='closest'  # Ensure the closest point is always shown in hover
        )
    
    return fig, {'display': 'block'}, {'display': 'block'}

# Run the app
if __name__ == '__main__':
    app.run(debug=True)