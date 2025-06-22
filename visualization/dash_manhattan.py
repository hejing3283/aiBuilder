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
        '_carriers', '_variants', '_variant_ids'
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
        'pval': 'log10_p_value' # The -log10 transformed p-value
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
        'test_specific_carriers', 'test_specific_variants', 'test_specific_variant_ids'
    ]
    
    # Ensure all columns in column_order exist before reordering
    existing_columns = [col for col in column_order if col in final_data_long.columns]
    final_data_long = final_data_long[existing_columns]

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
    
    # Reformat button
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
        )
    ], style={'textAlign': 'center'}),
    
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
     Input('clicked-points', 'data'),
     Input('btn-reformat', 'n_clicks')],
    [State('upload-data', 'filename')]
)
def update_plot(contents, threshold, point_size, test_type, clicked_points, n_clicks, filename):
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
        
        # Create intermediate long format
        intermediate_df = create_intermediate_data(processed_df)
        if intermediate_df.empty:
            return None, html.Div("No valid data in intermediate format"), "", "Error: No valid data"
        
        status = "Creating plot..."
        
        # Create file info message
        phenotype = intermediate_df['phenotype'].iloc[0] if 'phenotype' in intermediate_df.columns and not intermediate_df['phenotype'].empty else 'N/A'
        num_genes = intermediate_df['gene'].nunique() if 'gene' in intermediate_df.columns else 0
        num_tests = len(intermediate_df)
        num_chroms = intermediate_df['chromosome'].nunique() if 'chromosome' in intermediate_df.columns else 0

        file_info = html.Div([
            f"File: {filename} | ",
            f"Genes: {num_genes:,} | ",
            f"Tests: {num_tests:,} | ",
            f"Chromosomes: {num_chroms} | ",
            f"Phenotype: {phenotype}"
        ])
        
        # Filter data for selected test type (already in long format)
        plot_df = intermediate_df[intermediate_df['TEST_TYPE'] == test_type].copy()
        
        if plot_df.empty:
            return None, html.Div(f"No data available for {test_type}"), file_info, "Error: No data for selected test type"
        
        # Create Manhattan plot using Plotly
        manhattan_plot = go.Figure()
        
        # Calculate cumulative positions for each chromosome
        chrom_positions = {}
        cumulative_pos = 0
        for chrom in sorted(plot_df['chromosome'].unique()):
            chrom_max_pos = plot_df[plot_df['chromosome'] == chrom]['position'].max()
            if pd.isna(chrom_max_pos):
                chrom_max_pos = 0
            chrom_positions[chrom] = cumulative_pos
            cumulative_pos += chrom_max_pos 
            if chrom_max_pos > 0:
                cumulative_pos += 1e7 # A reasonable gap, adjust as needed

        # Add points for each chromosome
        for chrom in sorted(plot_df['chromosome'].unique()):
            chrom_data = plot_df[plot_df['chromosome'] == chrom].copy()
            # Add chromosome offset to positions
            chrom_offset = chrom_positions.get(chrom, 0)
            x_positions = chrom_data['position'] + chrom_offset
            
            # Use log10_p_value for y-axis
            y_values = chrom_data['log10_p_value']
            
            # Create customdata for hover template
            custom_data = np.column_stack((
                chrom_data['p_value'],             # 0: raw p-value
                chrom_data['effect_size'],         # 1: effect_size
                chrom_data['fdr'],                 # 2: fdr
                chrom_data['total_variants'],      # 3: total_variants (gene-level)
                chrom_data['total_carriers'],      # 4: total_carriers (gene-level)
                chrom_data['test_specific_variants'], # 5: test_specific_variants
                chrom_data['test_specific_carriers'], # 6: test_specific_carriers
                chrom_data['chromosome'],           # 7: chromosome
                chrom_data['position'],             # 8: position
                chrom_data['phenotype']             # 9: phenotype
            ))

            # Create hover template - uses columns from the long format df
            hover_template = (
                "<b>Gene:</b> %{text}<br>" +
                "<b>Test Type:</b> " + test_type.replace('_', ' ').title() + "<br>" +
                "<b>Chromosome:</b> %{customdata[7]}<br>" +
                "<b>Position:</b> %{customdata[8]:,.0f}<br>" +
                "<b>Phenotype:</b> %{customdata[9]}<br>" +
                "<b>P-value:</b> %{customdata[0]:.2e}<br>" +
                "<b>Effect Size:</b> %{customdata[1]:.3f}<br>" +
                "<b>FDR:</b> %{customdata[2]:.2e}<br>" +
                "<b>Total Variants (Gene):</b> %{customdata[3]:,}<br>" +
                "<b>Total Carriers (Gene):</b> %{customdata[4]:,}<br>" +
                "<b>Test-specific Variants:</b> %{customdata[5]:,}<br>" +
                "<b>Test-specific Carriers:</b> %{customdata[6]:,}<br>" +
                "<extra></extra>"
            )
            
            # Add main scatter plot
            manhattan_plot.add_trace(go.Scatter(
                x=x_positions,
                y=y_values, 
                mode='markers',
                name=f'Chr{chrom}',
                text=chrom_data['gene'],
                customdata=custom_data,
                marker=dict(
                    size=point_size,
                    color='skyblue' if chrom % 2 == 0 else 'royalblue'
                ),
                hovertemplate=hover_template
            ))
            
            # Add labels for significant points
            significant_points = chrom_data[chrom_data['log10_p_value'] >= threshold].copy()
            if not significant_points.empty:
                manhattan_plot.add_trace(go.Scatter(
                    x=significant_points['position'] + chrom_offset,
                    y=significant_points['log10_p_value'],
                    mode='markers+text',
                    text=significant_points['gene'],
                    textposition="top center",
                    textfont=dict(size=10),
                    marker=dict(size=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Add labels for clicked points
            if clicked_points:
                clicked_data = chrom_data[chrom_data['gene'].isin(clicked_points)].copy()
                if not clicked_data.empty:
                    manhattan_plot.add_trace(go.Scatter(
                        x=clicked_data['position'] + chrom_offset,
                        y=clicked_data['log10_p_value'],
                        mode='markers+text',
                        text=clicked_data['gene'],
                        textposition="top center",
                        textfont=dict(size=10),
                        marker=dict(size=0),
                        showlegend=False,
                        hoverinfo='skip'
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
        
        # Update layout
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
        unique_chroms_in_plot_df = sorted(plot_df['chromosome'].unique())
        for chrom in unique_chroms_in_plot_df:
            if chrom in chrom_positions:
                chrom_data_for_mean = plot_df[plot_df['chromosome'] == chrom]['position']
                if not chrom_data_for_mean.empty:
                    x_ticks.append(chrom_positions[chrom] + chrom_data_for_mean.mean())
                    x_labels.append(str(chrom))
        
        manhattan_plot.update_xaxes(
            ticktext=x_labels,
            tickvals=x_ticks,
            tickangle=0
        )
        
        return intermediate_df.to_dict('records'), html.Div([
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
        
        # The processed_data is now the intermediate long format
        # Filter by gene and TEST_TYPE to get the specific row
        gene_data = None
        for item in processed_data:
            if item.get('gene') == gene_name and item.get('TEST_TYPE') == test_type:
                gene_data = item
                break
                
        if not gene_data:
            return html.Div()
        
        # Create annotation panel content
        annotation_content = [
            html.H4("Gene Details"),
            html.Div([
                html.Strong("Gene: "), html.Span(str(gene_data.get('gene', 'N/A'))),
                html.Br(),
                html.Strong("Chromosome: "), html.Span(str(gene_data.get('chromosome', 'N/A'))),
                html.Br(),
                html.Strong("Position: "), html.Span(str(gene_data.get('position', 'N/A'))),
                html.Br(),
                html.Strong(f"{test_type.replace('_', ' ').title()} Burden Test:"),
                html.Br(),
                html.Strong("P-value: "), 
                html.Span(f"{gene_data.get('p_value', 0):.2e}"), 
                html.Br(),
                html.Strong("Effect Size: "), 
                html.Span(f"{gene_data.get('effect_size', 0):.3f}"),
                html.Br(),
                html.Strong("FDR: "), 
                html.Span(f"{gene_data.get('fdr', 1):.2e}"),
                html.Br(),
                html.Strong("Total Variants (Gene): "), 
                html.Span(str(gene_data.get('total_variants', 0))),
                html.Br(),
                html.Strong("Total Carriers (Gene): "), 
                html.Span(str(gene_data.get('total_carriers', 0))),
                html.Br(),
                html.Strong("Test-specific Variants: "), 
                html.Span(str(gene_data.get('test_specific_variants', 0))),
                html.Br(),
                html.Strong("Test-specific Carriers: "), 
                html.Span(str(gene_data.get('test_specific_carriers', 0)))
            ])
        ]
        
        # Add variant IDs if available
        if gene_data.get('test_specific_variant_ids'):
            try:
                variant_ids = str(gene_data['test_specific_variant_ids']).split(',')
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
                print(f"Error processing variants in annotation panel: {str(e)}")
        
        # Add phenotype information if available
        if gene_data.get('phenotype'):
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