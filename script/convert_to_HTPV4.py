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

def convert_to_htpv4(df, phenotype_col):
    """
    Convert the DRAGON output DataFrame to the HTPV4 format.
    The HTPV4 format typically includes: ID, chromosome, position, allele1, allele2, and phenotypes.
    """
    print("Converting to HTPV4 format...")
    
    # Check for required columns
    if 'gene_name' not in df.columns:
        raise ValueError("Input file is missing the required 'gene_name' column.")
    
    all_variants_data = []

    # Dynamically find all variant_ids columns
    variant_id_cols = [col for col in df.columns if col.endswith('/burden_test/variant_ids')]
    if not variant_id_cols:
        raise ValueError("No '/burden_test/variant_ids' columns found in the input file.")

    print(f"Found variant ID columns: {variant_id_cols}")

    for index, row in df.iterrows():
        gene_name = row['gene_name']
        phenotype_values = row[phenotype_col] if phenotype_col and phenotype_col in row else 'NA'

        # Use a set to store unique variants for this gene
        variants_for_gene = set()

        for col_name in variant_id_cols:
            if pd.notna(row[col_name]):
                variants = str(row[col_name]).split(',')
                variants_for_gene.update(variants)
        
        for var_id in variants_for_gene:
            try:
                # Format: "chromosome:position:reference:alternative"
                parts = var_id.split(':')
                if len(parts) == 4:
                    chrom, pos, ref, alt = parts
                    
                    # Create an entry for the HTPV4 format
                    variant_entry = {
                        'ID': var_id,
                        'chromosome': chrom,
                        'position': pos,
                        'allele1': ref,
                        'allele2': alt,
                        phenotype_col: phenotype_values if phenotype_col else 'NA'
                    }
                    all_variants_data.append(variant_entry)
            except ValueError:
                print(f"Warning: Skipping malformed variant ID '{var_id}' for gene '{gene_name}'.")
                continue

    if not all_variants_data:
        print("Warning: No valid variants were found to convert.")
        return pd.DataFrame()

    htpv4_df = pd.DataFrame(all_variants_data)

    # Reorder columns to have phenotype last
    cols = ['ID', 'chromosome', 'position', 'allele1', 'allele2']
    if phenotype_col:
        cols.append(phenotype_col)
    
    htpv4_df = htpv4_df[cols]
    
    print(f"Successfully converted {len(htpv4_df)} variants to HTPV4 format.")
    return htpv4_df

def main():
    parser = argparse.ArgumentParser(description="Convert DRAGON output files to REGENIE HTPV4 format.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input DRAGON output file (can be .txt, .gz, or .zip).")
    parser.add_argument("--output", "-o", required=True, help="Path to the output HTPV4 formatted file.")
    
    args = parser.parse_args()

    try:
        # Read the input file
        print(f"Reading input file: {args.input}")
        dragon_df = read_dragon_output(args.input)
        
        # Identify the phenotype column
        phenotype_col_name = get_phenotype_column(dragon_df)
        if phenotype_col_name:
            print(f"Identified phenotype column: '{phenotype_col_name}'")
        else:
            print("No distinct phenotype column identified. Phenotype data will be 'NA'.")

        # Convert to HTPV4 format
        htpv4_df = convert_to_htpv4(dragon_df, phenotype_col_name)

        # Save the output file
        if not htpv4_df.empty:
            htpv4_df.to_csv(args.output, sep=' ', index=False)
            print(f"Successfully saved HTPV4 formatted data to: {args.output}")

    except (FileNotFoundError, ValueError) as e:
        print(str(e))
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main() 