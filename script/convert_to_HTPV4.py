import pandas as pd
import argparse
import gzip
import zipfile
import os

def read_dragon_output(file_path):
    """Read a DRAGON output file, supporting .gz, .zip, or uncompressed."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found.")

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')
    elif file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as z:
            # Assuming there's only one file in the zip archive
            file_in_zip = z.namelist()[0]
            with z.open(file_in_zip) as f:
                df = pd.read_csv(f, sep='\t')
    else:
        df = pd.read_csv(file_path, sep='\t')
    return df

def get_phenotype_column(df):
    """
    Identifies the phenotype column.
    Assumes the last column is the phenotype if it's not a standard DRAGON output column.
    """
    # List of known non-phenotype columns that could be last
    known_cols = {'gene_name', 'CHROM'}
    # Dynamically find burden test columns
    burden_cols = [col for col in df.columns if '/burden_test/' in col]
    known_cols.update(burden_cols)
    
    last_col = df.columns[-1]
    if last_col not in known_cols:
        return last_col
    return None

def convert_to_gene_summary(df, phenotype_col):
    """
    Convert the DRAGON output DataFrame to a per-gene summary.
    """
    print("Converting to per-gene summary format...")
    
    if 'gene_name' not in df.columns:
        raise ValueError("Input file is missing the required 'gene_name' column.")
    
    all_genes_data = []

    variant_id_cols = [col for col in df.columns if col.endswith('/burden_test/variant_ids')]
    if not variant_id_cols:
        raise ValueError("No '/burden_test/variant_ids' columns found in the input file.")

    print(f"Found variant ID columns: {variant_id_cols}")

    for index, row in df.iterrows():
        gene_name = row['gene_name']
        chrom = row.get('CHROM', 'NA')
        phenotype_values = row.get(phenotype_col, 'NA') if phenotype_col else 'NA'

        variants_for_gene = set()
        for col_name in variant_id_cols:
            if pd.notna(row[col_name]):
                variants = str(row[col_name]).split(',')
                variants_for_gene.update(variants)
        
        if variants_for_gene:
            gene_entry = {
                'gene_name': gene_name,
                'chromosome': chrom,
                'variants': ','.join(sorted(list(variants_for_gene)))
            }
            if phenotype_col:
                gene_entry[phenotype_col] = phenotype_values
            all_genes_data.append(gene_entry)

    if not all_genes_data:
        print("Warning: No valid genes with variants were found to convert.")
        return pd.DataFrame()

    gene_df = pd.DataFrame(all_genes_data)

    cols = ['gene_name', 'chromosome']
    if phenotype_col:
        cols.append(phenotype_col)
    cols.append('variants')
    
    gene_df = gene_df[cols]
    
    print(f"Successfully converted {len(gene_df)} genes.")
    return gene_df

def main():
    parser = argparse.ArgumentParser(
        description="Convert DRAGON output files to a per-gene summary file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python %(prog)s --input your_dragon_output.txt.gz --output your_gene_summary.tsv
"""
    )
    parser.add_argument("--input", "-i", required=True, help="Path to the input DRAGON output file (can be .txt, .gz, or .zip).")
    parser.add_argument("--output", "-o", help="Path to the output per-gene summary file. Defaults to the input filename with a .htpv4.tsv extension.")
    
    args = parser.parse_args()

    output_path = args.output
    if not output_path:
        input_path = args.input
        # Create a default output path in the same directory as the input
        dirname = os.path.dirname(input_path)
        basename = os.path.basename(input_path)
        
        # Remove known extensions to get the base filename
        if basename.endswith('.gz'):
            basename = basename[:-3]
        elif basename.endswith('.zip'):
            basename = basename[:-4]
        
        if basename.endswith('.txt') or basename.endswith('.tsv'):
            basename = os.path.splitext(basename)[0]
            
        output_filename = basename + '.htpv4.tsv'
        output_path = os.path.join(dirname, output_filename)
        print(f"Output path not specified. Defaulting to: {output_path}")

    try:
        print(f"Reading input file: {args.input}")
        dragon_df = read_dragon_output(args.input)
        
        phenotype_col_name = get_phenotype_column(dragon_df)
        if phenotype_col_name:
            print(f"Identified phenotype column: '{phenotype_col_name}'")
        else:
            print("No distinct phenotype column identified.")

        gene_summary_df = convert_to_gene_summary(dragon_df, phenotype_col_name)

        if not gene_summary_df.empty:
            gene_summary_df.to_csv(output_path, sep='\t', index=False)
            print(f"Successfully saved per-gene summary data to: {output_path}")

    except (FileNotFoundError, ValueError) as e:
        print(str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main() 