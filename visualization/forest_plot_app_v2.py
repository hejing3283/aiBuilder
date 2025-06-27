import pandas as pd
import dash
from dash import html, dcc, Input, Output, callback, State
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

# New imports for forestplot
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import forestplot as fp

# --- Data Processing Functions (copied from forest_plot_app.py) ---

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
    
    processed_df = pd.DataFrame()
    
    if 'gene_name' not in df.columns:
        raise ValueError("Required column 'gene_name' is missing")
    if 'CHROM' not in df.columns:
        raise ValueError("Required column 'CHROM' is missing")
    
    processed_df = pd.DataFrame(index=range(len(df)))
    processed_df['GENE'] = df['gene_name'].astype(str)
    
    last_column = df.columns[-1]
    processed_df['phenotype'] = df[last_column].fillna('N/A')
    
    try:
        chrom_data = df['CHROM'].astype(str).str.replace('chr', '', case=False)
        chrom_map = {'X': '23', 'Y': '24', 'MT': '25', 'M': '25'}
        chrom_data = chrom_data.replace(chrom_map)
        processed_df['CHR'] = pd.to_numeric(chrom_data, errors='coerce')
        
        if processed_df['CHR'].isna().any():
            print("Warning: Some chromosome values could not be converted to numbers")
            processed_df['CHR'] = processed_df['CHR'].fillna(0)
    except Exception as e:
        raise ValueError(f"Error processing chromosome data: {str(e)}")
    
    variant_positions = {}
    for test_type in ['deleterious_coding', 'ptv', 'missense_prioritized', 'synonymous']:
        variant_ids_col = f'{test_type}/burden_test/variant_ids'
        if variant_ids_col in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row[variant_ids_col]):
                    positions = []
                    for var_id in str(row[variant_ids_col]).split(','):
                        try:
                            pos = int(var_id.split(':')[1])
                            positions.append(pos)
                        except (IndexError, ValueError):
                            continue
                    if positions:
                        gene = row['gene_name']
                        if gene not in variant_positions or min(positions) < variant_positions[gene]:
                            variant_positions[gene] = min(positions)
    
    processed_df['BP'] = processed_df['GENE'].map(variant_positions)
    
    burden_test_cols = [col for col in df.columns if '/burden_test/' in col]
    burden_data = {}
    for col in burden_test_cols:
        parts = col.split('/burden_test/')
        if len(parts) == 2:
            test_type, statistic = parts
            if test_type not in burden_data:
                burden_data[test_type] = {}
            burden_data[test_type][statistic] = col

    for test_type in burden_data:
        for statistic, col_name in burden_data[test_type].items():
            if col_name in df.columns:
                if statistic == 'pvalue':
                     processed_df[f'{test_type}_raw_pval'] = df[col_name].fillna(1)
                     processed_df[f'{test_type}_pval'] = -np.log10(df[col_name].fillna(1))
                elif statistic == 'beta':
                     processed_df[f'{test_type}_beta'] = df[col_name].fillna(0)
                else:
                     processed_df[f'{test_type}_{statistic}'] = df[col_name].fillna(0)

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

    test_types = []
    for col in processed_df.columns:
        if col.endswith('_beta'):
            test_type = col.replace('_beta', '')
            if f'{test_type}_raw_pval' in processed_df.columns:
                test_types.append(test_type)
    
    if not test_types:
        raise ValueError("No test types with both beta and p-value data found")

    forest_data = []
    for test_type in test_types:
        beta_col = f'{test_type}_beta'
        pval_col = f'{test_type}_raw_pval'
        
        test_data = processed_df[['GENE', beta_col, pval_col]].copy()
        test_data = test_data.dropna(subset=[beta_col, pval_col])
        
        if not test_data.empty:
            test_data['TEST_TYPE'] = test_type
            test_data.rename(columns={
                'GENE': 'gene',
                beta_col: 'effect_size',
                pval_col: 'p_value'
            }, inplace=True)
            
            test_data['p_value'] = test_data['p_value'].clip(1e-300, 1 - 1e-16)
            test_data['z_score'] = np.abs(norm.ppf(test_data['p_value'] / 2))
            test_data['std_error'] = np.abs(test_data['effect_size'] / test_data['z_score'])
            
            # Replace infinite values with NaN so forestplot can handle them
            test_data.replace([np.inf, -np.inf], np.nan, inplace=True)

            test_data['ci_lower'] = test_data['effect_size'] - 1.96 * test_data['std_error']
            test_data['ci_upper'] = test_data['effect_size'] + 1.96 * test_data['std_error']
            
            forest_data.append(test_data)
    
    if not forest_data:
        raise ValueError("No valid data found for forest plot")
    
    forest_df = pd.concat(forest_data, ignore_index=True)
    return forest_df

# --- Dash App ---

app = dash.Dash(__name__, title="Forest Plot Visualizer v2")
server = app.server

app.layout = html.Div([
    html.H1("Forest Plot (forestplot package)", style={'textAlign': 'center'}),
    
    dcc.Store(id='processed-forest-data-v2'),

    # --- Controls ---
    html.Div([
        dcc.Upload(
            id='upload-data-forest-v2',
            children=html.Button('Upload DRAGON Output File'),
            style={'display': 'inline-block', 'marginRight': '20px'}
        ),
        dcc.Dropdown(
            id='test-type-dropdown-forest-v2',
            placeholder="Select Burden Tests",
            multi=True,
            style={'display': 'inline-block', 'width': '300px', 'marginRight': '20px'}
        ),
        dcc.Dropdown(
            id='gene-dropdown-forest-v2',
            placeholder="Select Genes",
            multi=True,
            style={'display': 'inline-block', 'width': '300px', 'marginRight': '20px'}
        ),
        dcc.Dropdown(
            id='sort-order-dropdown-v2',
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
    html.Div(id='upload-status-v2', style={'textAlign': 'center', 'padding': '10px'}),

    # --- Plot ---
    dcc.Loading(
        id="loading-forest-v2",
        type="circle",
        children=html.Div(id='forest-plot-container-v2', children=html.Div("Please upload a file and make selections to view the plot."))
    ),
])

# --- Callbacks ---

@callback(
    [Output('processed-forest-data-v2', 'data'),
     Output('test-type-dropdown-forest-v2', 'options'),
     Output('gene-dropdown-forest-v2', 'options'),
     Output('upload-status-v2', 'children')],
    [Input('upload-data-forest-v2', 'contents')],
    [State('upload-data-forest-v2', 'filename')],
    prevent_initial_call=True
)
def process_file_for_forest_plot_v2(contents, filename):
    if contents is None:
        return dash.no_update

    try:
        raw_df = read_compressed_file(contents, filename)
        processed_df = process_dragon_output(raw_df)
        forest_df = process_data_for_forest_plot(processed_df)
        
        test_types = forest_df['TEST_TYPE'].unique()
        test_type_options = [{'label': tt.replace('_', ' ').title(), 'value': tt} for tt in test_types]

        genes = sorted(forest_df['gene'].unique())
        gene_options = [{'label': gene, 'value': gene} for gene in genes]
        
        success_message = f"✅ Upload complete! Loaded {len(forest_df)} records for {len(genes)} genes across {len(test_types)} test types."
        return forest_df.to_dict('records'), test_type_options, gene_options, success_message
        
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        return None, [], [], error_message

@callback(
    Output('forest-plot-container-v2', 'children'),
    [Input('processed-forest-data-v2', 'data'),
     Input('test-type-dropdown-forest-v2', 'value'),
     Input('gene-dropdown-forest-v2', 'value'),
     Input('sort-order-dropdown-v2', 'value')]
)
def update_forest_plot_v2(processed_data, selected_test_types, selected_genes, sort_order):
    if not processed_data or not selected_test_types or not selected_genes:
        return html.Div("Please upload a file and make selections to view the plot.")

    df = pd.DataFrame(processed_data)
    
    plot_df = df[(df['TEST_TYPE'].isin(selected_test_types)) & (df['gene'].isin(selected_genes))].copy()
    
    if plot_df.empty:
        return html.Div("No data available for the current selection.")

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
            plot_df = plot_df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    # Prepare data for forestplot package
    plot_df = plot_df.rename(columns={
        'effect_size': 'mean',
        'ci_lower': 'lower',
        'ci_upper': 'upper'
    })
    
    # Create y-labels
    plot_df['ylabel'] = plot_df.apply(lambda row: f"{row['gene']} ({row['TEST_TYPE']})", axis=1)

    # Plotting with forestplot
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.5)))
    fp.forestplot(plot_df,
                  ax=ax,
                  varlabel='ylabel',
                  estimate='mean',
                  lower='lower',
                  upper='upper',
                  color='TEST_TYPE',
                  xticklabelsize=12,
                  yticklabelsize=12)

    # Convert plot to SVG image for Dash
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    svg_data = buf.getvalue()
    
    encoded_svg = base64.b64encode(svg_data.encode('utf-8')).decode('utf-8')
    
    return html.Img(src=f"data:image/svg+xml;base64,{encoded_svg}")


if __name__ == '__main__':
    app.run_server(debug=True, port=8052) 