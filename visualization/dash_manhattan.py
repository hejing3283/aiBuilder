## pip install -r visualization/requirements.txt
## python visualization/dash_manhattan.py
## Open your web browser and go to http://localhost:8050
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
    # test_types = ['deleterious_coding', 'ptv', 'missense_prioritized', 'synonymous'] # Removed hardcoded list

    # Initialize columns for each test type based on dynamically found test types
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
        if processed_df[col].isna().any():
            print(f"Warning: Column {col} contains missing values, filling with appropriate defaults")
            if col == 'CHR':
                processed_df[col] = processed_df[col].fillna(0)
            elif col == 'BP':
                processed_df[col] = processed_df[col].fillna(0)
            elif col == 'GENE':
                processed_df[col] = processed_df[col].fillna('Unknown')
    
    print("--- Finished process_dragon_output ---")
    return processed_df

def create_intermediate_data(processed_df):
    """Create an intermediate data format by melting and pivoting."""
    print("\n=== Starting create_intermediate_data (Melting/Pivoting) ===")
    print("Input DataFrame shape:", processed_df.shape)
    print("Input DataFrame columns:", processed_df.columns.tolist())

    if processed_df.empty:
        print("Processed DataFrame is empty, returning empty intermediate data.")
        return pd.DataFrame()

    # Columns that describe the gene, not the burden test result, to keep as ID variables
    id_vars = ['GENE', 'CHR', 'BP', 'phenotype']

    # Collect all burden test related columns and their types
    # These are the columns that will be melted
    burden_stat_suffixes = [
        '_raw_pval', '_pval', '_beta', '_fdr', 
        '_carriers', '_variants', '_variant_ids',
        '_n_carriers_case', '_n_carriers_control'
    ]
    
    # Dynamically find columns to melt
    cols_to_melt = []
    for col in processed_df.columns:
        for suffix in burden_stat_suffixes:
            if col.endswith(suffix):
                cols_to_melt.append(col)
                break 

    if not cols_to_melt:
        print("No burden test specific columns found for melting.")
        return pd.DataFrame()

    # Ensure all id_vars exist in processed_df
    for col in id_vars:
        if col not in processed_df.columns:
            print(f"Error: Required ID column '{col}' missing from processed_df.")
            return pd.DataFrame()

    # Melt the DataFrame
    print(f"Columns to melt: {cols_to_melt}")
    melted_df = processed_df.melt(id_vars=id_vars, value_vars=cols_to_melt, var_name='original_column', value_name='value')
    print("Melted DataFrame shape:", melted_df.shape)

    # Extract TEST_TYPE and statistic type from 'original_column'
    def parse_original_column(col_name):
        for suffix in burden_stat_suffixes:
            if col_name.endswith(suffix):
                test_type = col_name[:-len(suffix)]
                statistic_type = suffix[1:] # remove leading underscore
                return test_type, statistic_type
        return None, None

    melted_df[['TEST_TYPE', 'statistic']] = melted_df['original_column'].apply(lambda x: pd.Series(parse_original_column(x)))
    
    # Drop rows where parsing failed or value is NaN (important for pivot)
    melted_df.dropna(subset=['TEST_TYPE', 'statistic'], inplace=True)
    melted_df['value'] = pd.to_numeric(melted_df['value'], errors='coerce') # Ensure value is numeric
    melted_df.dropna(subset=['value'], inplace=True) # Drop rows where value conversion failed

    print("Melted DataFrame with TEST_TYPE and statistic columns head:\n", melted_df.head())

    # Pivot the DataFrame to get statistics as columns
    pivot_index = ['GENE', 'CHR', 'BP', 'phenotype', 'TEST_TYPE']
    
    # Ensure all pivot_index columns are available in melted_df
    for col in pivot_index:
        if col not in melted_df.columns:
            print(f"Error: Column {col} missing in melted_df for pivoting index.")
            return pd.DataFrame() 
    
    # Pivot table
    final_data_long = melted_df.pivot_table(index=pivot_index, columns='statistic', values='value', aggfunc='first').reset_index()
    
    # Rename columns to desired format
    final_data_long.rename(columns={
        'raw_pval': 'p_value',
        'beta': 'effect_size',
        'fdr': 'fdr',
        'carriers': 'test_specific_carriers',
        'variants': 'test_specific_variants',
        'variant_ids': 'test_specific_variant_ids',
        'pval': 'log10_p_value', # The -log10 transformed p-value
        'n_carriers_case': 'carriers_case',
        'n_carriers_control': 'carriers_control'
    }, inplace=True)
    
    # --- Calculate gene_common_info and merge ---    
    # Create a dictionary of common gene information from the original processed_df
    common_info_agg = processed_df.groupby('GENE').agg(
        chromosome=('CHR', 'first'), # Take first chromosome for gene
        position=('BP', 'first'),    # Take first BP for gene
        phenotype=('phenotype', 'first') # Take first phenotype for gene
    ).reset_index()

    # Initialize all_variant_ids and total_carriers
    common_info_agg['all_variant_ids'] = ''
    common_info_agg['total_carriers'] = 0
    common_info_agg['total_variants'] = 0

    # Collect all variant IDs and carriers across test types for each gene
    for idx, gene_row in common_info_agg.iterrows():
        gene = gene_row['GENE']
        original_gene_rows = processed_df[processed_df['GENE'] == gene]
        
        all_variants_for_gene = set()
        all_carriers_for_gene = set()

        # Iterate through the dynamically found test types
        test_types_in_processed_df = [col.split('/')[0] for col in processed_df.columns if '/burden_test/pvalue' in col]
        for tt in test_types_in_processed_df:
            variant_ids_col = f'{tt}_variant_ids'
            carriers_col = f'{tt}_carriers'
            
            if variant_ids_col in original_gene_rows.columns:
                for var_ids_str in original_gene_rows[variant_ids_col].dropna().astype(str):
                    all_variants_for_gene.update(var_ids_str.split(','))
            
            if carriers_col in original_gene_rows.columns:
                for num_carriers in original_gene_rows[carriers_col].dropna():
                    all_carriers_for_gene.add(num_carriers)
        
        common_info_agg.loc[idx, 'total_variants'] = len(all_variants_for_gene)
        common_info_agg.loc[idx, 'total_carriers'] = max(all_carriers_for_gene) if all_carriers_for_gene else 0
        common_info_agg.loc[idx, 'all_variant_ids'] = ','.join(sorted(all_variants_for_gene)) if all_variants_for_gene else ''

    # Merge common gene info back into the long format DataFrame
    final_data_long = pd.merge(final_data_long, common_info_agg[['GENE', 'total_variants', 'total_carriers', 'all_variant_ids']], on='GENE', how='left')

    # Rename 'GENE' to 'gene', 'CHR' to 'chromosome', 'BP' to 'position' for consistency
    final_data_long.rename(columns={'GENE': 'gene', 'CHR': 'chromosome', 'BP': 'position'}, inplace=True)


    print("\nFinal data columns (long format):", final_data_long.columns.tolist())
    print("Final data shape (long format):", final_data_long.shape)
    
    # Sort by gene and test type
    final_data_long = final_data_long.sort_values(['gene', 'TEST_TYPE'])
    
    # Reorder columns for better readability
    column_order = [
        'gene', 'chromosome', 'position', 'phenotype', 'TEST_TYPE',
        'p_value', 'log10_p_value', 'effect_size', 'fdr', 
        'total_variants', 'total_carriers', 'all_variant_ids',
        'test_specific_carriers', 'carriers_case', 'carriers_control',
        'test_specific_variants', 'test_specific_variant_ids'
    ]
    
    # Ensure all columns in column_order exist before reordering
    existing_columns = [col for col in column_order if col in final_data_long.columns]
    final_data_long = final_data_long[existing_columns]

    print("\nColumns in final reformatted data:", final_data_long.columns.tolist())

    # Save intermediate data to file for inspection
    try:
        output_path = os.path.join(os.path.dirname(__file__), 'intermediate_data_long_format.csv')
        final_data_long.to_csv(output_path, index=False)
        print(f"\nSaved intermediate data (long format) to: {output_path}")
        print("First few rows of the saved long format data:\n", final_data_long.head())
    except Exception as e:
        print(f"\nError saving intermediate data (long format): {str(e)}")
        print("Current working directory:", os.getcwd())

    print("=== Finished create_intermediate_data ===")
    return final_data_long

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

    # Reformat and Download buttons
    html.Div([
        html.Button(
            "Reformat Data",
            id="btn-reformat",
            style={
                'margin': '10px',
                'padding': '10px 20px',
                'backgroundColor': '#FFC107',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
        ),
        dcc.Download(id='download-dataframe-csv'),
        html.Button(
            "Download Formatted Data",
            id="btn-download",
            style={
                'margin': '10px',
                'padding': '10px 20px',
                'backgroundColor': '#4CAF50',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer'
            }
        )
    ], style={'textAlign': 'center'}),

    # Test type selector
    html.Div([
        html.Label("Select Burden Test Type(s):"),
        dcc.Dropdown(
            id='test-type-dropdown',
            style={'width': '50%'},
            placeholder="Upload a file to see options",
            multi=True
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
        ], style={'width': '45%', 'display': 'inline-block', 'margin': '10px'}),
        
        # Add store for clicked points
        dcc.Store(id='clicked-points', data=[])
    ], style={'textAlign': 'center', 'margin': '20px'}),

    # Main content area with plot and annotation panel
    html.Div([
        # Manhattan plot
        html.Div([
            html.Div(id='manhattan-container', children=html.Div("Please upload a file to see the plot."))
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

# Callback for file processing
@callback(
    [Output('processed-data', 'data'),
     Output('file-info', 'children'),
     Output('progress-status', 'children'),
     Output('test-type-dropdown', 'options'),
     Output('test-type-dropdown', 'value')],
    [Input('upload-data', 'contents'),
     Input('btn-reformat', 'n_clicks')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def process_uploaded_file(contents, n_clicks, filename):
    ctx = dash.callback_context
    if not ctx.triggered or contents is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    try:
        df = read_compressed_file(contents, filename)
        if df.empty:
            return None, "Error: Uploaded file is empty", "Error", [], None

        processed_df = process_dragon_output(df)
        if processed_df.empty:
            return None, "Error: No valid data after processing", "Error", [], None

        intermediate_df = create_intermediate_data(processed_df)
        if intermediate_df.empty:
            return None, "Error: No valid data in intermediate format", "Error", [], None

        test_types_found = intermediate_df['TEST_TYPE'].unique()
        dropdown_options = [{'label': tt.replace('_', ' ').title(), 'value': tt} for tt in test_types_found]
        # Default to all test types selected
        default_test_types = [opt['value'] for opt in dropdown_options]
        
        phenotype = intermediate_df['phenotype'].iloc[0] if 'phenotype' in intermediate_df.columns and not intermediate_df['phenotype'].empty else 'N/A'
        file_info = f"File: {filename} | Genes: {intermediate_df['gene'].nunique():,} | Tests: {len(intermediate_df):,} | Chromosomes: {intermediate_df['chromosome'].nunique()} | Phenotype: {phenotype}"
        
        return intermediate_df.to_dict('records'), file_info, "Processing complete!", dropdown_options, default_test_types

    except Exception as e:
        return None, f"Error processing file: {e}", "Error", [], None

# Callback for updating the plot
@callback(
    Output('manhattan-container', 'children'),
    [Input('processed-data', 'data'),
     Input('threshold-slider', 'value'),
     Input('test-type-dropdown', 'value'),
     Input('clicked-points', 'data')]
)
def update_manhattan_plot(processed_data, threshold, test_types, clicked_points): # test_type is now test_types (plural)
    if not processed_data or not test_types:
        return html.Div("Upload a file and select at least one test type to view the plot.")

    intermediate_df = pd.DataFrame(processed_data)
    # Filter for all selected test types
    plot_df = intermediate_df[intermediate_df['TEST_TYPE'].isin(test_types)].copy()
    
    if plot_df.empty:
        return html.Div(f"No data available for the selected test type(s).")

    manhattan_plot = go.Figure()

    # Calculate cumulative positions for all chromosomes across all selected tests
    chrom_positions = {}
    cumulative_pos = 0
    for chrom in sorted(plot_df['chromosome'].unique()):
        chrom_max_pos = plot_df[plot_df['chromosome'] == chrom]['position'].max()
        chrom_positions[chrom] = cumulative_pos
        cumulative_pos += (chrom_max_pos if pd.notna(chrom_max_pos) else 0) + 1e7

    # Plot each selected test type as a separate trace for the legend
    for test_type in test_types:
        test_type_df = plot_df[plot_df['TEST_TYPE'] == test_type]
        if test_type_df.empty:
            continue

        x_positions = test_type_df.apply(lambda row: row['position'] + chrom_positions.get(row['chromosome'], 0), axis=1)
        symbols = test_type_df['effect_size'].apply(lambda es: 'triangle-up' if es > 1 else ('triangle-down' if es < 1 else 'circle')).tolist()
        
        # Use the previous chromosome-based color scheme
        point_colors = test_type_df['chromosome'].apply(lambda c: 'skyblue' if c % 2 == 0 else 'royalblue').tolist()
        
        custom_data = np.column_stack((
            test_type_df['p_value'], test_type_df['effect_size'], test_type_df['fdr'],
            test_type_df['total_variants'], test_type_df['total_carriers'],
            test_type_df['test_specific_variants'], test_type_df['test_specific_carriers'],
            test_type_df['chromosome'], test_type_df['position'], test_type_df['phenotype']
        ))
        
        hover_template = "<b>Gene:</b> %{text}<br><b>Test Type:</b> " + test_type.replace('_', ' ').title() + "<br><b>P-value:</b> %{customdata[0]:.2e}<br><b>Effect Size:</b> %{customdata[1]:.3f}<extra></extra>"

        manhattan_plot.add_trace(go.Scatter(
            x=x_positions, y=test_type_df['log10_p_value'], mode='markers',
            name=test_type.replace('_', ' ').title(), # Name for the legend
            text=test_type_df['gene'], customdata=custom_data, hovertemplate=hover_template,
            marker=dict(
                size=8, # Increased default point size
                color=point_colors,
                symbol=symbols
            )
        ))

    # Add labels for significant points from all selected tests
    significant_points = plot_df[plot_df['log10_p_value'] >= threshold]
    if not significant_points.empty:
        sig_x_pos = significant_points.apply(lambda row: row['position'] + chrom_positions.get(row['chromosome'], 0), axis=1)
        manhattan_plot.add_trace(go.Scatter(
            x=sig_x_pos, y=significant_points['log10_p_value'],
            mode='text', text=significant_points['gene'], textposition="top center",
            textfont=dict(size=10), showlegend=False, hoverinfo='none'
        ))

    # Add labels for clicked points from all selected tests
    if clicked_points:
        clicked_data = plot_df[plot_df['gene'].isin(clicked_points)]
        if not clicked_data.empty:
            clicked_x_pos = clicked_data.apply(lambda row: row['position'] + chrom_positions.get(row['chromosome'], 0), axis=1)
            manhattan_plot.add_trace(go.Scatter(
                x=clicked_x_pos, y=clicked_data['log10_p_value'],
                mode='text', text=clicked_data['gene'], textposition="top center",
                textfont=dict(size=10), showlegend=False, hoverinfo='none'
            ))

    manhattan_plot.add_hline(y=threshold, line_dash="dash", line_color="red")
    manhattan_plot.update_layout(
        title=f"Manhattan Plot",
        xaxis_title="Genomic Position", yaxis_title="-log10(p-value)",
        showlegend=True, hovermode='closest', plot_bgcolor='white', paper_bgcolor='white',
        width=1200, height=600, margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            tickvals=[chrom_positions[c] + plot_df[plot_df['chromosome']==c]['position'].mean() for c in sorted(plot_df['chromosome'].unique()) if not plot_df[plot_df['chromosome']==c].empty],
            ticktext=[str(c) for c in sorted(plot_df['chromosome'].unique()) if not plot_df[plot_df['chromosome']==c].empty]
        )
    )
    
    return dcc.Graph(figure=manhattan_plot, id='manhattan-graph')

@callback(
    Output('annotation-panel', 'children'),
    [Input('manhattan-graph', 'hoverData')],
    [State('processed-data', 'data'),
     State('test-type-dropdown', 'value')]
)
def update_annotation_panel(hover_data, processed_data, test_type):
    if not hover_data or not processed_data or not test_type:
        return html.Div()
    
    gene_name = hover_data['points'][0]['text']
    gene_data = next((item for item in processed_data if item.get('gene') == gene_name and item.get('TEST_TYPE') == test_type), None)
    
    if not gene_data:
        return html.Div(f"No details found for {gene_name} with test type {test_type}")

    print(f"\nAnnotation panel data for {gene_name} ({test_type}):\n{gene_data}\n")

    details_children = [
        html.Strong("Gene: "), html.Span(str(gene_data.get('gene', 'N/A'))), html.Br(),
        html.Strong("Chromosome: "), html.Span(str(gene_data.get('chromosome', 'N/A'))), html.Br(),
        html.Strong("Position: "), html.Span(str(gene_data.get('position', 'N/A'))), html.Br(),
        html.Strong(f"{test_type.replace('_', ' ').title()} Burden Test:"), html.Br(),
        html.Strong("P-value: "), html.Span(f"{gene_data.get('p_value', 0):.2e}"), html.Br(),
        html.Strong("Effect Size: "), html.Span(f"{gene_data.get('effect_size', 0):.3f}"), html.Br(),
        html.Strong("FDR: "), html.Span(f"{gene_data.get('fdr', 1):.2e}"), html.Br(),
        html.Strong("Total Variants (Gene): "), html.Span(str(gene_data.get('total_variants', 0))), html.Br(),
        html.Strong("Total Carriers (Gene): "), html.Span(str(gene_data.get('total_carriers', 0))), html.Br(),
        html.Strong("Test-specific Variants: "), html.Span(str(gene_data.get('test_specific_variants', 0))), html.Br(),
        html.Strong("Test-specific Carriers: "), html.Span(str(gene_data.get('test_specific_carriers', 0))),
    ]

    # Conditionally add case/control carriers
    if 'carriers_case' in gene_data and pd.notna(gene_data['carriers_case']):
        details_children.append(
            html.Div([html.Strong("Case Carriers: "), html.Span(f"{int(gene_data.get('carriers_case', 0))}")], style={'marginLeft': '15px'})
        )
    if 'carriers_control' in gene_data and pd.notna(gene_data['carriers_control']):
        details_children.append(
            html.Div([html.Strong("Control Carriers: "), html.Span(f"{int(gene_data.get('carriers_control', 0))}")], style={'marginLeft': '15px'})
        )

    annotation_content = [
        html.H4("Gene Details"),
        html.Div(details_children)
    ]

    # Add variant IDs if available
    variant_ids_key = 'test_specific_variant_ids'
    if variant_ids_key in gene_data and gene_data[variant_ids_key] and pd.notna(gene_data[variant_ids_key]):
        variant_ids = str(gene_data[variant_ids_key]).split(',')
        variant_list = []
        for var_id in variant_ids:
            try:
                chrom, pos, ref, alt = var_id.split(':')
                variant_list.append(f"{chrom}:{pos} {ref}→{alt}")
            except (ValueError, IndexError):
                variant_list.append(var_id)
        
        if variant_list:
            annotation_content.append(html.Div([
                html.Br(),
                html.Strong("Variants: "), html.Br(),
                html.Div([html.Span(var, style={'display': 'block'}) for var in variant_list], style={'marginLeft': '20px'})
            ]))

    # Add phenotype information if available
    if 'phenotype' in gene_data and gene_data['phenotype']:
        annotation_content.append(html.Div([
            html.Br(),
            html.Strong("Phenotype: "),
            html.Span(str(gene_data['phenotype']))
        ]))
    
    return html.Div(annotation_content)

@callback(
    Output('clicked-points', 'data'),
    [Input('manhattan-graph', 'clickData')],
    [State('clicked-points', 'data')]
)
def update_clicked_points(click_data, current_clicked):
    if click_data:
        clicked_gene = click_data['points'][0]['text']
        if clicked_gene not in current_clicked:
            current_clicked.append(clicked_gene)
    return current_clicked

@callback(
    Output("download-dataframe-csv", "data"),
    [Input("btn-download", "n_clicks")],
    [State('processed-data', 'data')],
    prevent_initial_call=True,
)
def download_data(n_clicks, processed_data):
    if n_clicks is None or not processed_data:
        return dash.no_update
    
    df_to_download = pd.DataFrame(processed_data)
    return dcc.send_data_frame(df_to_download.to_csv, "dragon_burden_results_long_format.csv", index=False)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050) 