import pandas as pd
import dash
from dash import html, dcc, Input, Output, callback, State
import plotly.graph_objects as go
import base64
import io
import numpy as np
import gzip
import zipfile
import tempfile
import os
from scipy.stats import norm
import time
from dash.exceptions import PreventUpdate
import plotly.io as pio

# --- Data Processing Functions (reused from Manhattan plot app) ---

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
    
    # Dynamically find and process burden test results
    burden_test_cols = [col for col in df.columns if '/burden_test/' in col]
    print(f"Detected burden test columns: {burden_test_cols}")

    # Group columns by burden type and statistic
    burden_data = {}
    for col in burden_test_cols:
        parts = col.split('/burden_test/')
        if len(parts) == 2:
            test_type = parts[0]
            statistic = parts[1]

            if test_type not in burden_data:
                burden_data[test_type] = {}
            burden_data[test_type][statistic] = col

    test_types_found = list(burden_data.keys())
    print(f"Identified test types: {test_types_found}")

    # Process different burden test results
    for test_type in test_types_found:
        print(f"\nProcessing detected test type: {test_type}:")
        # P-value
        pval_col_raw = burden_data[test_type].get('pvalue')
        if pval_col_raw and pval_col_raw in df.columns:
            # Store raw p-values
            processed_df[f'{test_type}_raw_pval'] = df[pval_col_raw].fillna(1)
            # Convert p-values to -log10 scale for plotting
            processed_df[f'{test_type}_pval'] = -np.log10(df[pval_col_raw].fillna(1))
            print(f"  - Processed p-value for {test_type}")
        else:
            print(f"  - P-value column not found or invalid for {test_type}")

        # Effect size (beta)
        beta_col_raw = burden_data[test_type].get('beta')
        if beta_col_raw and beta_col_raw in df.columns:
            processed_df[f'{test_type}_beta'] = df[beta_col_raw].fillna(0)
            print(f"  - Processed beta for {test_type}")

        # FDR
        fdr_col_raw = burden_data[test_type].get('fdr')
        if fdr_col_raw and fdr_col_raw in df.columns:
            processed_df[f'{test_type}_fdr'] = df[fdr_col_raw].fillna(1)
            print(f"  - Processed FDR for {test_type}")

        # Number of carriers
        carriers_col_raw = burden_data[test_type].get('n_carriers')
        if carriers_col_raw and carriers_col_raw in df.columns:
            processed_df[f'{test_type}_carriers'] = df[carriers_col_raw].fillna(0)
            print(f"  - Processed n_carriers for {test_type}")

        # Add case/control carriers
        carriers_case_col_raw = burden_data[test_type].get('n_carriers_case')
        print(f"  - Looking for case carriers column: {test_type}/burden_test/n_carriers_case -> Found: {carriers_case_col_raw}")
        if carriers_case_col_raw and carriers_case_col_raw in df.columns:
            processed_df[f'{test_type}_n_carriers_case'] = df[carriers_case_col_raw].fillna(0)
            print(f"    - Processed n_carriers_case for {test_type}")

        carriers_control_col_raw = burden_data[test_type].get('n_carriers_control')
        print(f"  - Looking for control carriers column: {test_type}/burden_test/n_carriers_control -> Found: {carriers_control_col_raw}")
        if carriers_control_col_raw and carriers_control_col_raw in df.columns:
            processed_df[f'{test_type}_n_carriers_control'] = df[carriers_control_col_raw].fillna(0)
            print(f"    - Processed n_carriers_control for {test_type}")

        # Number of variants
        variants_col_raw = burden_data[test_type].get('n_variants')
        if variants_col_raw and variants_col_raw in df.columns:
            processed_df[f'{test_type}_variants'] = df[variants_col_raw].fillna(0)
            print(f"  - Processed n_variants for {test_type}")

        # Add variant IDs
        variant_ids_col_raw = burden_data[test_type].get('variant_ids')
        if variant_ids_col_raw and variant_ids_col_raw in df.columns:
            processed_df[f'{test_type}_variant_ids'] = df[variant_ids_col_raw].fillna('')
            print(f"  - Processed variant_ids for {test_type}")

    print("\nFinal Processed DataFrame columns:", processed_df.columns.tolist())
    
    # Ensure all required columns exist and have valid data
    required_columns = ['GENE', 'CHR', 'BP']
    for col in required_columns:
        if col not in processed_df.columns:
            raise ValueError(f"Required column {col} is missing from the processed data")
    
    return processed_df

def process_data_for_forest_plot(processed_df):
    """
    Processes the processed DRAGON output into a format suitable for the forest plot.
    Calculates Standard Error and Confidence Intervals.
    """
    if processed_df.empty:
        raise ValueError("Input dataframe is empty")
    if 'GENE' not in processed_df.columns:
        raise ValueError("Required column 'GENE' is missing")

    # Find all test types that have beta and p-value data
    test_types = []
    for col in processed_df.columns:
        if col.endswith('_beta'):
            test_type = col.replace('_beta', '')
            if f'{test_type}_raw_pval' in processed_df.columns:
                test_types.append(test_type)
    
    if not test_types:
        raise ValueError("No test types with both beta and p-value data found")

    # Create forest plot data
    forest_data = []
    for test_type in test_types:
        beta_col = f'{test_type}_beta'
        pval_col = f'{test_type}_raw_pval'
        
        # Get data for this test type
        test_data = processed_df[['GENE', beta_col, pval_col]].copy()
        test_data = test_data.dropna(subset=[beta_col, pval_col])
        
        if not test_data.empty:
            test_data['TEST_TYPE'] = test_type
            test_data.rename(columns={
                'GENE': 'gene',
                beta_col: 'effect_size',
                pval_col: 'p_value'
            }, inplace=True)
            
            # Calculate Standard Error and 95% Confidence Intervals
            # Avoid p-values of 0 or 1 for Z-score calculation
            test_data['p_value'] = test_data['p_value'].clip(1e-300, 1 - 1e-16)
            test_data['z_score'] = np.abs(norm.ppf(test_data['p_value'] / 2))
            test_data['std_error'] = np.abs(test_data['effect_size'] / test_data['z_score'])
            test_data['ci_lower'] = test_data['effect_size'] - 1.96 * test_data['std_error']
            test_data['ci_upper'] = test_data['effect_size'] + 1.96 * test_data['std_error']
            
            forest_data.append(test_data)
    
    if not forest_data:
        raise ValueError("No valid data found for forest plot")
    
    forest_df = pd.concat(forest_data, ignore_index=True)
    return forest_df

# --- Dash App ---

app = dash.Dash(__name__, title="Forest Plot Visualizer")
server = app.server

# Add custom CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .upload-status {
                font-weight: bold;
                color: #333;
                margin-top: 10px;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
            .upload-error {
                color: #d32f2f;
                background-color: #ffebee;
                border: 1px solid #ffcdd2;
            }
            .upload-success {
                color: #388e3c;
                background-color: #e8f5e8;
                border: 1px solid #c8e6c9;
            }
            .upload-info {
                color: #1976d2;
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
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

app.layout = html.Div([
    html.H1("Forest Plot for Gene Effect Sizes", style={'textAlign': 'center'}),
    
    dcc.Store(id='processed-forest-data'),
    dcc.Store(id='original-processed-data'),  # Store the original processed data for n_carriers

    # --- Controls ---
    html.Div([
        # File Upload
        dcc.Upload(
            id='upload-data-forest',
            children=html.Button('Upload DRAGON Output File'),
            style={'display': 'inline-block', 'marginRight': '20px'}
        ),
        # Test Type Dropdown
        dcc.Dropdown(
            id='test-type-dropdown-forest',
            placeholder="Select Burden Tests",
            multi=True,
            style={'display': 'inline-block', 'width': '300px', 'marginRight': '20px'}
        ),
        # Gene Search Dropdown
        dcc.Dropdown(
            id='gene-dropdown-forest',
            placeholder="Select Genes",
            multi=True,
            style={'display': 'inline-block', 'width': '300px', 'marginRight': '20px'}
        ),
        # Remove gene-test-order-dropdown, add sort-order-dropdown
        dcc.Dropdown(
            id='sort-order-dropdown',
            options=[
                {'label': 'Gene', 'value': 'gene'},
                {'label': 'Test Type', 'value': 'TEST_TYPE'},
                {'label': 'P-value (ascending)', 'value': 'p_value_asc'},
                {'label': 'P-value (descending)', 'value': 'p_value_desc'},
                {'label': 'Effect Size (ascending)', 'value': 'effect_size_asc'},
                {'label': 'Effect Size (descending)', 'value': 'effect_size_desc'},
            ],
            multi=True,
            placeholder="Select and order variables to sort rows",
            style={'display': 'inline-block', 'width': '400px'}
        ),
    ], style={'textAlign': 'center', 'padding': '20px'}),

    # --- Progress and Status ---
    html.Div([
        dcc.Loading(
            id="loading-upload",
            type="circle",
            children=html.Div(id='upload-status', className='upload-status', style={'display': 'none'})
        )
    ], style={'margin': '20px'}),

    # --- Plot ---
    dcc.Loading(
        id="loading-forest",
        type="circle",
        children=html.Div([
            html.Div(id='forest-plot-container', children=html.Div("Please upload a file and make selections to view the plot.")),
            html.Div([
                html.Button("Download SVG", id="download-svg-btn", style={"marginRight": "10px"}),
                dcc.Download(id="download-svg"),
                html.Button("Download PDF", id="download-pdf-btn", style={"marginRight": "10px"}),
                dcc.Download(id="download-pdf"),
            ], style={"textAlign": "center", "marginTop": "20px"})
        ])
    ),
])

# --- Callbacks ---

# 1. Process uploaded file and populate dropdowns
@callback(
    [Output('processed-forest-data', 'data'),
     Output('original-processed-data', 'data'),
     Output('test-type-dropdown-forest', 'options'),
     Output('gene-dropdown-forest', 'options'),
     Output('upload-status', 'children'),
     Output('upload-status', 'className'),
     Output('upload-status', 'style')],
    [Input('upload-data-forest', 'contents')],
    [State('upload-data-forest', 'filename')],
    prevent_initial_call=True
)
def process_file_for_forest_plot(contents, filename):
    if contents is None:
        return dash.no_update

    try:
        # Show status with info style
        status_style = {'display': 'block'}
        
        # Update status - Starting file read
        raw_df = read_compressed_file(contents, filename)
        
        # Update status - File read complete, start processing
        processed_df = process_dragon_output(raw_df)
        
        # Update status - Data processing complete, prepare forest plot data
        forest_df = process_data_for_forest_plot(processed_df)
        
        test_types = forest_df['TEST_TYPE'].unique()
        test_type_options = [{'label': tt.replace('_', ' ').title(), 'value': tt} for tt in test_types]

        genes = sorted(forest_df['gene'].unique())
        gene_options = [{'label': gene, 'value': gene} for gene in genes]
        
        # Update status - Complete
        success_message = f"✅ Upload complete! Loaded {len(forest_df)} records for {len(genes)} genes across {len(test_types)} test types."
        return forest_df.to_dict('records'), processed_df.to_dict('records'), test_type_options, gene_options, success_message, 'upload-status upload-success', status_style
        
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        return None, None, [], [], error_message, 'upload-status upload-error', {'display': 'block'}

# 2. Update plot based on selections
@callback(
    Output('forest-plot-container', 'children'),
    [Input('processed-forest-data', 'data'),
     Input('original-processed-data', 'data'),
     Input('test-type-dropdown-forest', 'value'),
     Input('gene-dropdown-forest', 'value'),
     Input('sort-order-dropdown', 'value')]
)
def update_forest_plot(processed_data, original_processed_data, selected_test_types, selected_genes, sort_order):
    if not processed_data or not selected_test_types or not selected_genes:
        return html.Div("Please upload a file and make selections to view the plot.")

    df = pd.DataFrame(processed_data)
    original_df = pd.DataFrame(original_processed_data)
    
    # Filter data based on selections
    plot_df = df[(df['TEST_TYPE'].isin(selected_test_types)) & (df['gene'].isin(selected_genes))].copy()
    
    if plot_df.empty:
        return html.Div("No data available for the current selection.")

    # --- Flexible sorting logic ---
    if sort_order:
        sort_cols = []
        ascending = []
        for col in sort_order:
            if col == 'gene':
                sort_cols.append('gene')
                ascending.append(True)
            elif col == 'TEST_TYPE':
                sort_cols.append('TEST_TYPE')
                ascending.append(True)
            elif col == 'p_value_asc':
                sort_cols.append('p_value')
                ascending.append(True)
            elif col == 'p_value_desc':
                sort_cols.append('p_value')
                ascending.append(False)
            elif col == 'effect_size_asc':
                sort_cols.append('effect_size')
                ascending.append(True)
            elif col == 'effect_size_desc':
                sort_cols.append('effect_size')
                ascending.append(False)
        if sort_cols:
            plot_df = plot_df.sort_values(by=sort_cols, ascending=ascending)

    # Get additional data for annotations
    carrier_data = []
    case_carrier_data = []
    control_carrier_data = []
    
    for test_type in selected_test_types:
        carriers_col = f'{test_type}_carriers'
        case_carriers_col = f'{test_type}_n_carriers_case'
        control_carriers_col = f'{test_type}_n_carriers_control'
        
        if carriers_col in original_df.columns:
            test_carriers = original_df[original_df['GENE'].isin(selected_genes)][['GENE', carriers_col]].copy()
            test_carriers['TEST_TYPE'] = test_type
            carrier_data.append(test_carriers)
        
        if case_carriers_col in original_df.columns:
            test_case_carriers = original_df[original_df['GENE'].isin(selected_genes)][['GENE', case_carriers_col]].copy()
            test_case_carriers['TEST_TYPE'] = test_type
            case_carrier_data.append(test_case_carriers)
        
        if control_carriers_col in original_df.columns:
            test_control_carriers = original_df[original_df['GENE'].isin(selected_genes)][['GENE', control_carriers_col]].copy()
            test_control_carriers['TEST_TYPE'] = test_type
            control_carrier_data.append(test_control_carriers)

    # Create the forest plot
    fig = go.Figure()
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # For each test type, create a separate trace
    for i, test_type in enumerate(selected_test_types):
        test_data = plot_df[plot_df['TEST_TYPE'] == test_type].copy()
        if not test_data.empty:
            color = colors[i % len(colors)]
            
            # Create gene labels with test type suffix for multiple tests
            if len(selected_test_types) > 1:
                gene_labels = [f"{gene} ({test_type.replace('_', ' ').title()})" for gene in test_data['gene']]
            else:
                gene_labels = test_data['gene']
            
            fig.add_trace(go.Scatter(
                x=test_data['effect_size'],
                y=gene_labels,
                mode='markers',
                marker=dict(color=color, size=10),
                name=test_type.replace('_', ' ').title(),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=test_data['ci_upper'] - test_data['effect_size'],
                    arrayminus=test_data['effect_size'] - test_data['ci_lower'],
                    thickness=1.5,
                    width=5,
                    color=color
                ),
                hovertemplate="<b>%{y}</b><br>Effect Size: %{x:.3f}<br>P-value: %{customdata[0]:.2e}<extra></extra>",
                customdata=test_data[['p_value']]
            ))

    # Add text annotations for additional information
    if carrier_data:
        carriers_df = pd.concat(carrier_data, ignore_index=True)
        carriers_df = carriers_df.rename(columns={'GENE': 'gene'})
        
        case_carriers_df = None
        if case_carrier_data:
            case_carriers_df = pd.concat(case_carrier_data, ignore_index=True)
            case_carriers_df = case_carriers_df.rename(columns={'GENE': 'gene'})
        
        control_carriers_df = None
        if control_carrier_data:
            control_carriers_df = pd.concat(control_carrier_data, ignore_index=True)
            control_carriers_df = control_carriers_df.rename(columns={'GENE': 'gene'})
        
        # Add carrier information as text
        for i, test_type in enumerate(selected_test_types):
            test_data = plot_df[plot_df['TEST_TYPE'] == test_type].copy()
            if not test_data.empty:
                color = colors[i % len(colors)]
                
                # Create gene labels
                if len(selected_test_types) > 1:
                    gene_labels = [f"{gene} ({test_type.replace('_', ' ').title()})" for gene in test_data['gene']]
                else:
                    gene_labels = test_data['gene']
                
                # Get carrier counts for this test type
                test_carriers = carriers_df[carriers_df['TEST_TYPE'] == test_type]
                test_case_carriers = case_carriers_df[case_carriers_df['TEST_TYPE'] == test_type] if case_carriers_df is not None else None
                test_control_carriers = control_carriers_df[control_carriers_df['TEST_TYPE'] == test_type] if control_carriers_df is not None else None
                
                # Create separate text annotations for p-value and carrier counts
                pvalue_annotations = []
                carrier_annotations = []
                case_carrier_annotations = []
                control_carrier_annotations = []
                
                for idx, row in test_data.iterrows():
                    gene = row['gene']
                    p_val = row['p_value']
                    
                    # Get carrier counts
                    carrier_count = test_carriers[test_carriers['gene'] == gene][f'{test_type}_carriers'].iloc[0] if not test_carriers[test_carriers['gene'] == gene].empty else 'N/A'
                    
                    case_count = 'N/A'
                    if test_case_carriers is not None and not test_case_carriers[test_case_carriers['gene'] == gene].empty:
                        case_count = test_case_carriers[test_case_carriers['gene'] == gene][f'{test_type}_n_carriers_case'].iloc[0]
                    
                    control_count = 'N/A'
                    if test_control_carriers is not None and not test_control_carriers[test_control_carriers['gene'] == gene].empty:
                        control_count = test_control_carriers[test_control_carriers['gene'] == gene][f'{test_type}_n_carriers_control'].iloc[0]
                    
                    # Format text
                    p_text = f"p={p_val:.2e}" if p_val < 0.01 else f"p={p_val:.3f}"
                    carrier_text = str(int(carrier_count)) if carrier_count != 'N/A' and not pd.isna(carrier_count) else 'N/A'
                    case_text = str(int(case_count)) if case_count != 'N/A' and not pd.isna(case_count) else 'N/A'
                    control_text = str(int(control_count)) if control_count != 'N/A' and not pd.isna(control_count) else 'N/A'
                    
                    pvalue_annotations.append(p_text)
                    carrier_annotations.append(carrier_text)
                    case_carrier_annotations.append(case_text)
                    control_carrier_annotations.append(control_text)
                
                # Calculate positions for the columns
                max_effect = test_data['effect_size'].max()
                min_effect = test_data['effect_size'].min()
                effect_range = max_effect - min_effect
                
                # Position columns with much more spacing to avoid overlap with error bars and between text columns
                pvalue_x_pos = max_effect + (effect_range * 1.0)  # P-value column
                carrier_x_pos = max_effect + (effect_range * 1.5)  # Total carriers column - increased spacing
                case_x_pos = max_effect + (effect_range * 2.0)     # Case carriers column - increased spacing
                control_x_pos = max_effect + (effect_range * 2.5)  # Control carriers column - increased spacing
                
                # Add p-value text trace
                fig.add_trace(go.Scatter(
                    x=[pvalue_x_pos] * len(test_data),
                    y=gene_labels,
                    mode='text',
                    text=pvalue_annotations,
                    textposition='middle right',
                    textfont=dict(size=10, color=color),
                    name=f"{test_type.replace('_', ' ').title()} (P-value)",
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add total carrier count text trace
                fig.add_trace(go.Scatter(
                    x=[carrier_x_pos] * len(test_data),
                    y=gene_labels,
                    mode='text',
                    text=carrier_annotations,
                    textposition='middle right',
                    textfont=dict(size=10, color=color),
                    name=f"{test_type.replace('_', ' ').title()} (Total)",
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add case carrier count text trace
                fig.add_trace(go.Scatter(
                    x=[case_x_pos] * len(test_data),
                    y=gene_labels,
                    mode='text',
                    text=case_carrier_annotations,
                    textposition='middle right',
                    textfont=dict(size=10, color=color),
                    name=f"{test_type.replace('_', ' ').title()} (Case)",
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Add control carrier count text trace
                fig.add_trace(go.Scatter(
                    x=[control_x_pos] * len(test_data),
                    y=gene_labels,
                    mode='text',
                    text=control_carrier_annotations,
                    textposition='middle right',
                    textfont=dict(size=10, color=color),
                    name=f"{test_type.replace('_', ' ').title()} (Control)",
                    showlegend=False,
                    hoverinfo='skip'
                ))

    # Add a vertical line at zero effect
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")

    # Determine title
    if len(selected_test_types) > 1:
        title = f"Forest Plot for Multiple Burden Tests"
    else:
        title = f"Forest Plot for {selected_test_types[0].replace('_', ' ').title()}"

    fig.update_layout(
        title=title,
        xaxis_title="Effect Size (Beta) and 95% Confidence Interval",
        yaxis_title="Gene",
        height=max(400, len(plot_df) * 30), # Dynamic height based on number of data points
        showlegend=len(selected_test_types) > 1,
        plot_bgcolor='white',
        yaxis=dict(autorange="reversed"), # Show largest effect at the top
        xaxis=dict(range=[None, None])  # Allow x-axis to extend for text
    )

    return dcc.Graph(figure=fig)

# --- Download Callbacks ---

@callback(
    Output("download-svg", "data"),
    Input("download-svg-btn", "n_clicks"),
    State('forest-plot-container', 'children'),
    prevent_initial_call=True
)
def download_svg(n_clicks, graph_children):
    if not n_clicks:
        raise PreventUpdate

    # The 'children' prop of the container will be a dict representing the dcc.Graph component
    if isinstance(graph_children, dict) and graph_children.get('type') == 'Graph' and 'props' in graph_children:
        figure_data = graph_children['props'].get('figure')
        if figure_data:
            fig = go.Figure(figure_data)
            svg_bytes = pio.to_image(fig, format='svg')
            return dict(content=svg_bytes.decode(), filename="forest_plot.svg") # Decode for text-based formats
            
    return dash.no_update

@callback(
    Output("download-pdf", "data"),
    Input("download-pdf-btn", "n_clicks"),
    State('forest-plot-container', 'children'),
    prevent_initial_call=True
)
def download_pdf(n_clicks, graph_children):
    if not n_clicks:
        raise PreventUpdate

    # The 'children' prop of the container will be a dict representing the dcc.Graph component
    if isinstance(graph_children, dict) and graph_children.get('type') == 'Graph' and 'props' in graph_children:
        figure_data = graph_children['props'].get('figure')
        if figure_data:
            fig = go.Figure(figure_data)
            pdf_bytes = pio.to_image(fig, format='pdf')
            return dcc.send_bytes(pdf_bytes, "forest_plot.pdf")

    return dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True, port=8051) 