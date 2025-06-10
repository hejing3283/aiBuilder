## pip install -r visualization/requirements.txt
## python visualization/dash_manhattan.py
##Open your web browser and go to http://localhost:8050
## Upload your DRAGON output file with burden test results

import pandas as pd
import dash
from dash import html, dcc, Input, Output, callback
import dash_bio as dashbio
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import base64
import io
import numpy as np
import gzip
import zipfile
import tempfile
import os

# Initialize the Dash app
app = dash.Dash(__name__)

def read_compressed_file(content, filename):
    """Read data from various file formats including compressed files."""
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(decoded)
        temp_file.flush()
        
        try:
            if filename.endswith('.gz'):
                with gzip.open(temp_file.name, 'rt') as f:
                    df = pd.read_csv(f, sep='\t')
            elif filename.endswith('.zip'):
                with zipfile.ZipFile(temp_file.name) as z:
                    file_list = z.namelist()
                    if not file_list:
                        raise ValueError("Empty zip file")
                    with z.open(file_list[0]) as f:
                        df = pd.read_csv(f, sep='\t')
            else:
                df = pd.read_csv(temp_file.name, sep='\t')
                
            return df
            
        finally:
            os.unlink(temp_file.name)

def process_dragon_output(df):
    """Process the DRAGON output dataframe with burden test results."""
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Create a new dataframe with the required columns
    processed_df = pd.DataFrame()
    
    # Basic gene information - ensure we have valid data
    if 'gene_name' not in df.columns:
        raise ValueError("Required column 'gene_name' is missing")
    if 'CHROM' not in df.columns:
        raise ValueError("Required column 'CHROM' is missing")
    
    # Create a clean copy of the data
    processed_df = pd.DataFrame(index=range(len(df)))
    processed_df['GENE'] = df['gene_name'].astype(str)
    
    # Get phenotype from the last column
    last_column = df.columns[-1]
    processed_df['phenotype'] = df[last_column].fillna('N/A')
    
    # Handle chromosome data more carefully
    try:
        # First, try to convert chromosome numbers directly
        chrom_data = df['CHROM'].astype(str).str.replace('chr', '', case=False)
        # Handle special cases like 'X', 'Y', 'MT'
        chrom_map = {'X': '23', 'Y': '24', 'MT': '25', 'M': '25'}
        chrom_data = chrom_data.replace(chrom_map)
        # Convert to numeric, invalid values will become NaN
        processed_df['CHR'] = pd.to_numeric(chrom_data, errors='coerce')
        
        # Check for any remaining NaN values
        if processed_df['CHR'].isna().any():
            print("Warning: Some chromosome values could not be converted to numbers")
            # Fill NaN values with a placeholder (e.g., 0) to prevent errors
            processed_df['CHR'] = processed_df['CHR'].fillna(0)
    except Exception as e:
        raise ValueError(f"Error processing chromosome data: {str(e)}")
    
    # Get the leftmost variant position for each gene
    variant_positions = {}
    for test_type in ['deleterious_coding', 'ptv', 'missense_prioritized', 'synonymous']:
        variant_ids_col = f'{test_type}/burden_test/variant_ids'
        if variant_ids_col in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row[variant_ids_col]):
                    # Extract positions from variant IDs (format: "chromosome:position:reference:alternative")
                    positions = []
                    for var_id in str(row[variant_ids_col]).split(','):
                        try:
                            # Split by ':' and get the position (second element)
                            pos = int(var_id.split(':')[1])
                            positions.append(pos)
                        except (IndexError, ValueError):
                            continue
                    if positions:
                        gene = row['gene_name']
                        if gene not in variant_positions or min(positions) < variant_positions[gene]:
                            variant_positions[gene] = min(positions)
    
    # Add the leftmost position to the processed dataframe
    processed_df['BP'] = processed_df['GENE'].map(variant_positions)
    
    # Process different burden test results
    test_types = ['deleterious_coding', 'ptv', 'missense_prioritized', 'synonymous']
    
    # Initialize columns for each test type
    for test_type in test_types:
        # P-value
        pval_col = f'{test_type}/burden_test/pvalue'
        if pval_col in df.columns:
            # Convert p-values to -log10 scale for plotting
            processed_df[f'{test_type}_pval'] = -np.log10(df[pval_col].fillna(1))
            # Store raw p-values for display
            processed_df[f'{test_type}_raw_pval'] = df[pval_col].fillna(1)
        
        # Effect size (beta)
        beta_col = f'{test_type}/burden_test/beta'
        if beta_col in df.columns:
            processed_df[f'{test_type}_beta'] = df[beta_col].fillna(0)
        
        # FDR
        fdr_col = f'{test_type}/burden_test/fdr'
        if fdr_col in df.columns:
            processed_df[f'{test_type}_fdr'] = df[fdr_col].fillna(1)
        
        # Number of carriers
        carriers_col = f'{test_type}/burden_test/n_carriers'
        if carriers_col in df.columns:
            processed_df[f'{test_type}_carriers'] = df[carriers_col].fillna(0)
        
        # Number of variants
        variants_col = f'{test_type}/burden_test/n_variants'
        if variants_col in df.columns:
            processed_df[f'{test_type}_variants'] = df[variants_col].fillna(0)
        
        # Add variant IDs
        variant_ids_col = f'{test_type}/burden_test/variant_ids'
        if variant_ids_col in df.columns:
            processed_df[f'{test_type}_variant_ids'] = df[variant_ids_col].fillna('')
    
    # Ensure all required columns exist and have valid data
    required_columns = ['GENE', 'CHR', 'BP']
    for col in required_columns:
        if col not in processed_df.columns:
            raise ValueError(f"Required column {col} is missing from the processed data")
        if processed_df[col].isna().any():
            print(f"Warning: Column {col} contains missing values, filling with appropriate defaults")
            if col == 'CHR':
                processed_df[col] = processed_df[col].fillna(0)
            elif col == 'BP':
                processed_df[col] = processed_df[col].fillna(0)
            elif col == 'GENE':
                processed_df[col] = processed_df[col].fillna('Unknown')
    
    return processed_df

# Layout
app.layout = html.Div([
    html.H1("DRAGON Burden Test Manhattan Plot", 
            style={'textAlign': 'center', 'margin': '20px'}),
    
    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select DRAGON Output File'),
            html.Br(),
            html.Span('(Supports .txt, .gz, .zip files)', 
                     style={'fontSize': '12px', 'color': '#666'})
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
        }
    ),
    
    # Progress indicator
    html.Div([
        dcc.Loading(
            id="loading",
            type="circle",
            children=[
                html.Div(id='progress-status', style={'textAlign': 'center', 'margin': '10px'})
            ]
        )
    ]),
    
    # File info display
    html.Div(id='file-info', style={'margin': '10px', 'textAlign': 'center'}),
    
    # Test type selector
    html.Div([
        html.Label("Select Burden Test Type:"),
        dcc.Dropdown(
            id='test-type-dropdown',
            options=[
                {'label': 'Deleterious Coding', 'value': 'deleterious_coding'},
                {'label': 'PTV', 'value': 'ptv'},
                {'label': 'Missense Prioritized', 'value': 'missense_prioritized'},
                {'label': 'Synonymous', 'value': 'synonymous'}
            ],
            value='deleterious_coding',
            style={'width': '200px'}
        )
    ], style={'margin': '20px'}),
    
    # Controls
    html.Div([
        html.Div([
            html.Label("Significance Threshold (-log10):"),
            dcc.Slider(
                id='threshold-slider',
                min=1,
                max=20,
                step=0.1,
                value=5,
                marks={i: str(i) for i in range(1, 21, 2)}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
        
        html.Div([
            html.Label("Point Size:"),
            dcc.Slider(
                id='point-size-slider',
                min=1,
                max=10,
                step=0.5,
                value=5,
                marks={i: str(i) for i in range(1, 11)}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'margin': '10px'}),
        
        # Add store for clicked points
        dcc.Store(id='clicked-points', data=[])
    ], style={'margin': '20px'}),
    
    # Main content area with plot and annotation panel
    html.Div([
        # Manhattan plot
        html.Div([
            html.Div(id='manhattan-container', style={'width': '70%', 'display': 'inline-block'})
        ], style={'width': '100%', 'marginBottom': '20px'}),
        
        # Annotation panel
        html.Div([
            html.H3("Gene Details", style={'textAlign': 'center'}),
            html.Div(id='annotation-panel', style={
                'width': '25%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'marginLeft': '20px',
                'padding': '20px',
                'borderRadius': '5px'
            })
        ])
    ], style={'display': 'flex', 'margin': '20px', 'flexDirection': 'column'}),
    
    # Store for the current data
    dcc.Store(id='processed-data')
])

# Add CSS for progress bar
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .progress-bar {
                width: 100%;
                height: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
            }
            .progress-bar .progress-bar-fill {
                height: 100%;
                background-color: #4CAF50;
                transition: width 0.3s ease-in-out;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Update the callback to include progress updates
@callback(
    [Output('processed-data', 'data'),
     Output('manhattan-container', 'children'),
     Output('file-info', 'children'),
     Output('progress-status', 'children')],
    [Input('upload-data', 'contents'),
     Input('threshold-slider', 'value'),
     Input('point-size-slider', 'value'),
     Input('test-type-dropdown', 'value'),
     Input('clicked-points', 'data')],
    [State('upload-data', 'filename')]
)
def update_plot(contents, threshold, point_size, test_type, clicked_points, filename):
    if contents is None:
        return None, html.Div("Please upload a DRAGON output file"), "", ""
    
    try:
        # Read and process the file
        status = "Reading file..."
        df = read_compressed_file(contents, filename)
        if df.empty:
            return None, html.Div("The uploaded file is empty"), "", "Error: Empty file"
        
        status = "Processing data..."
        processed_df = process_dragon_output(df)
        if processed_df.empty:
            return None, html.Div("No valid data after processing"), "", "Error: No valid data"
        
        # Verify we have the required test type data
        required_cols = [f'{test_type}_pval', f'{test_type}_beta', f'{test_type}_fdr']
        missing_cols = [col for col in required_cols if col not in processed_df.columns]
        if missing_cols:
            return None, html.Div(f"Missing required columns for {test_type}: {', '.join(missing_cols)}"), "", "Error: Missing columns"
        
        status = "Creating plot..."
        
        # Create file info message
        phenotype = processed_df['phenotype'].iloc[0] if 'phenotype' in processed_df.columns else 'N/A'
        file_info = html.Div([
            f"File: {filename} | ",
            f"Genes: {len(processed_df):,} | ",
            f"Chromosomes: {processed_df['CHR'].nunique()} | ",
            f"Phenotype: {phenotype}"
        ])
        
        # Prepare data for Manhattan plot
        plot_data = {
            'CHR': processed_df['CHR'].values,
            'BP': processed_df['BP'].values,
            'P': processed_df[f'{test_type}_pval'].values,  # Use the selected test type's p-value
            'GENE': processed_df['GENE'].values,
            'SNP': processed_df['GENE'].values,
            'BETA': processed_df[f'{test_type}_beta'].values,
            'FDR': processed_df[f'{test_type}_fdr'].values,
            'CARRIERS': processed_df[f'{test_type}_carriers'].values,
            'VARIANTS': processed_df[f'{test_type}_variants'].values
        }
        
        # Create plot dataframe
        plot_df = pd.DataFrame(plot_data)
        
        # Remove any rows with missing or invalid data
        plot_df = plot_df.dropna(subset=['CHR', 'BP', 'P'])
        
        if plot_df.empty:
            return None, html.Div("No valid data points for plotting"), file_info, "Error: No valid data points"
        
        status = "Finalizing plot..."
        
        # Create Manhattan plot using Plotly
        manhattan_plot = go.Figure()
        
        # Calculate cumulative positions for each chromosome
        chrom_positions = {}
        cumulative_pos = 0
        for chrom in sorted(plot_df['CHR'].unique()):
            chrom_positions[chrom] = cumulative_pos
            cumulative_pos += plot_df[plot_df['CHR'] == chrom]['BP'].max()
        
        # Add points for each chromosome
        for chrom in sorted(plot_df['CHR'].unique()):
            chrom_data = plot_df[plot_df['CHR'] == chrom]
            # Add chromosome offset to positions
            x_positions = chrom_data['BP'] + chrom_positions[chrom]
            
            # Create hover template dynamically based on available data
            hover_template = (
                "<b>Gene:</b> %{text}<br>" +
                "<b>Chromosome:</b> " + str(chrom) + "<br>" +
                "<b>Position:</b> %{x:,.0f}<br>" +
                "<b>Phenotype:</b> " + str(chrom_data.get('phenotype', 'N/A')) + "<br>"
            )
            
            # Add test information only if p-values exist
            test_types = ['deleterious_coding', 'ptv', 'missense_prioritized', 'synonymous']
            for test_type in test_types:
                pval_col = f'{test_type}_raw_pval'
                if pval_col in processed_df.columns and not processed_df[pval_col].isna().all():
                    hover_template += (
                        f"<br><b>{test_type.replace('_', ' ').title()}:</b><br>" +
                        "P-value: " + f"{processed_df.loc[chrom_data.index, pval_col].iloc[0]:.2e}<br>" +
                        "Effect Size: " + f"{processed_df.loc[chrom_data.index, f'{test_type}_beta'].iloc[0]:.3f}<br>" +
                        "Carriers: " + f"{processed_df.loc[chrom_data.index, f'{test_type}_carriers'].iloc[0]:,}<br>" +
                        "Variants: " + f"{processed_df.loc[chrom_data.index, f'{test_type}_variants'].iloc[0]:,}<br>"
                    )
            
            hover_template += "<extra></extra>"
            
            # Add main scatter plot
            manhattan_plot.add_trace(go.Scatter(
                x=x_positions,
                y=chrom_data['P'],
                mode='markers',
                name=f'Chr{chrom}',
                text=chrom_data['GENE'],
                marker=dict(
                    size=point_size,
                    color='skyblue' if chrom % 2 == 0 else 'royalblue'
                ),
                hovertemplate=hover_template
            ))
            
            # Add labels for significant points (without hover)
            significant_points = chrom_data[chrom_data['P'] >= threshold]
            if not significant_points.empty:
                manhattan_plot.add_trace(go.Scatter(
                    x=significant_points['BP'] + chrom_positions[chrom],
                    y=significant_points['P'],
                    mode='markers+text',
                    text=significant_points['GENE'],
                    textposition="top center",
                    textfont=dict(size=10),
                    marker=dict(size=0),
                    showlegend=False,
                    hoverinfo='skip'  # Skip hover for label points
                ))
            
            # Add labels for clicked points (without hover)
            if clicked_points:
                clicked_data = chrom_data[chrom_data['GENE'].isin(clicked_points)]
                if not clicked_data.empty:
                    manhattan_plot.add_trace(go.Scatter(
                        x=clicked_data['BP'] + chrom_positions[chrom],
                        y=clicked_data['P'],
                        mode='markers+text',
                        text=clicked_data['GENE'],
                        textposition="top center",
                        textfont=dict(size=10),
                        marker=dict(size=0),
                        showlegend=False,
                        hoverinfo='skip'  # Skip hover for label points
                    ))
        
        # Add significance lines
        manhattan_plot.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red"
        )
        manhattan_plot.add_hline(
            y=threshold - 1,
            line_dash="dash",
            line_color="blue"
        )
        
        # Update layout with chromosome labels
        manhattan_plot.update_layout(
            title=f"Manhattan Plot - {test_type.replace('_', ' ').title()} Burden Test",
            xaxis_title="Genomic Position",
            yaxis_title="-log10(p-value)",
            showlegend=False,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=600,
            margin=dict(l=50, r=50, t=50, b=50),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black'
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black'
            )
        )
        
        # Add chromosome labels
        x_ticks = []
        x_labels = []
        for chrom in sorted(plot_df['CHR'].unique()):
            if chrom in chrom_positions:
                x_ticks.append(chrom_positions[chrom] + plot_df[plot_df['CHR'] == chrom]['BP'].mean())
                x_labels.append(str(chrom))
        
        manhattan_plot.update_xaxes(
            ticktext=x_labels,
            tickvals=x_ticks,
            tickangle=0
        )
        
        return processed_df.to_dict('records'), html.Div([
            dcc.Graph(
                figure=manhattan_plot,
                id='manhattan-graph',
                style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'}
            )
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'}), file_info, "Upload complete!"
    
    except Exception as e:
        error_message = f'Error processing file: {str(e)}'
        return None, html.Div(error_message), html.Div(error_message, style={'color': 'red'}), f"Error: {str(e)}"

# Update the callback for annotation panel
@callback(
    Output('annotation-panel', 'children'),
    [Input('manhattan-graph', 'hoverData'),
     Input('processed-data', 'data'),
     Input('test-type-dropdown', 'value')]
)
def update_annotation_panel(hover_data, processed_data, test_type):
    if not hover_data or not processed_data:
        return html.Div()
    
    try:
        # Get the hovered point data
        point = hover_data.get('points', [{}])[0]
        if not point:
            return html.Div()
            
        # Get gene name from text field
        gene_name = point.get('text', '')
        if not gene_name:
            return html.Div()
        
        # Find matching gene data
        gene_data = None
        for item in processed_data:
            if item.get('GENE') == gene_name:
                gene_data = item
                break
                
        if not gene_data:
            return html.Div()
        
        # Create annotation panel content
        annotation_content = [
            html.H4("Gene Details"),
            html.Div([
                html.Strong("Gene: "), html.Span(str(gene_data.get('GENE', 'N/A'))),
                html.Br(),
                html.Strong("Chromosome: "), html.Span(str(gene_data.get('CHR', 'N/A'))),
                html.Br(),
                html.Strong("Leftmost Position: "), html.Span(str(gene_data.get('BP', 'N/A'))),
                html.Br(),
                html.Strong(f"{test_type.replace('_', ' ').title()} Burden Test:"),
                html.Br(),
                html.Strong("P-value: "), 
                html.Span(f"{10**(-gene_data.get(f'{test_type}_pval', 0)):.2e}"),
                html.Br(),
                html.Strong("Effect Size: "), 
                html.Span(f"{gene_data.get(f'{test_type}_beta', 0):.3f}"),
                html.Br(),
                html.Strong("FDR: "), 
                html.Span(f"{gene_data.get(f'{test_type}_fdr', 1):.2e}"),
                html.Br(),
                html.Strong("Number of Carriers: "), 
                html.Span(str(gene_data.get(f'{test_type}_carriers', 0))),
                html.Br(),
                html.Strong("Number of Variants: "), 
                html.Span(str(gene_data.get(f'{test_type}_variants', 0)))
            ])
        ]
        
        # Add variant IDs if available
        variant_ids_key = f'{test_type}_variant_ids'
        if variant_ids_key in gene_data and gene_data[variant_ids_key]:
            try:
                variant_ids = str(gene_data[variant_ids_key]).split(',')
                variant_list = []
                for var_id in variant_ids:
                    try:
                        chrom, pos, ref, alt = var_id.split(':')
                        variant_list.append(f"{chrom}:{pos} {ref}→{alt}")
                    except ValueError:
                        variant_list.append(var_id)
                
                if variant_list:
                    annotation_content.append(html.Div([
                        html.Br(),
                        html.Strong("Variants: "),
                        html.Br(),
                        html.Div([
                            html.Span(var) for var in variant_list
                        ], style={'marginLeft': '20px'})
                    ]))
            except Exception as e:
                print(f"Error processing variants: {str(e)}")
        
        # Add phenotype information if available
        if 'phenotype' in gene_data and gene_data['phenotype']:
            annotation_content.append(html.Div([
                html.Br(),
                html.Strong("Phenotype: "),
                html.Span(str(gene_data['phenotype']))
            ]))
        
        return html.Div(annotation_content)
    
    except Exception as e:
        print(f"Error in annotation panel: {str(e)}")
        return html.Div()

# Add callback for clicked points
@callback(
    Output('clicked-points', 'data'),
    [Input('manhattan-graph', 'clickData')],
    [State('clicked-points', 'data')]
)
def update_clicked_points(click_data, current_clicked):
    if click_data is None:
        return current_clicked
    
    clicked_gene = click_data['points'][0]['text']
    if clicked_gene not in current_clicked:
        current_clicked.append(clicked_gene)
    return current_clicked

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 