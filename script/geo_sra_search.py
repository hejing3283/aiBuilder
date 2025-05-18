import pandas as pd
from Bio import Entrez
from datetime import datetime
import xml.etree.ElementTree as ET
import requests
import time
import json

Entrez.email = 'violet.hj@gmail.com'  # User's email address

# --- Test mode ---
test_mode = True  # Set to False to process all GSEs

# --- Search parameters ---
query = '(obesity) AND "Homo sapiens"[porgn:__txid9606]'
first_keyword = 'obesity'

# --- Search GEO ---
handle = Entrez.esearch(db='gds', term=query, retmax=20)
record = Entrez.read(handle)
gse_ids = [x for x in record['IdList']]
if test_mode:
    gse_ids = gse_ids[:1]
print(f'Found {len(gse_ids)} datasets')

allowed_types = [
    'Expression profiling by high throughput sequencing',
    'Genome binding/occupancy profiling by high throughput sequencing',
    'Genome variation profiling by high throughput sequencing',
    'Expression profiling by array'
]

results = []
all_metadata = []

def gse_has_tissue_attribute(gse_number):
    """Return True if any sample in the GSE has an attribute named 'tissue'."""
    # Fetch the list of GSMs for this GSE
    url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_number}&targ=gsm&form=text&view=brief'
    response = requests.get(url)
    gsm_ids = []
    for line in response.text.splitlines():
        if line.startswith('GSM'):
            gsm_ids.append(line.split()[0])
    for gsm_id in gsm_ids:
        xml_url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}&targ=self&form=xml&view=full'
        xml_response = requests.get(xml_url)
        try:
            root = ET.fromstring(xml_response.text)
            for ch in root.iter('Characteristics'):
                if 'tissue' in (ch.attrib.get('tag', '').lower() or ch.text.lower()):
                    return True
        except Exception:
            continue
    return False

for gse_id in gse_ids:
    # Fetch GEO summary
    summary = Entrez.esummary(db='gds', id=gse_id)
    meta = Entrez.read(summary)[0]
    print(f"Metadata for {meta.get('Accession', gse_id)}:")
    for key, value in meta.items():
        print(f"  {key}: {value}")
    print("-" * 40)
    all_metadata.append(meta)
    gse_number = meta.get('Accession', '')
    title = meta.get('title', '')
    samples = meta.get('n_samples', '')
    gds_type = meta.get('gdsType', '')
    if gds_type not in allowed_types:
        print(f"Skipping {gse_number} ({title}) - gdsType is '{gds_type}'")
        continue
    organism = meta.get('taxon', '') or meta.get('organism', '')
    if organism.lower() != 'homo sapiens':
        print(f"Skipping {gse_number} ({title}) - Organism is '{organism}'")
        continue
    # Filter for at least one sample with 'tissue' attribute
    if not gse_has_tissue_attribute(gse_number):
        print(f"Skipping {gse_number} ({title}) - No sample with 'tissue' attribute")
        continue
    pubmed_ids = meta.get('pubmed_ids', [])
    pubmed_link = ''
    authors = ''
    pub_year = ''
    if pubmed_ids:
        pubmed_id = pubmed_ids[0]
        pubmed_link = f"https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}"
        # Fetch PubMed metadata
        pubmed_handle = Entrez.efetch(db='pubmed', id=pubmed_id, retmode='xml')
        pubmed_record = Entrez.read(pubmed_handle)
        article = pubmed_record['PubmedArticle'][0]['MedlineCitation']['Article']
        # Get author names
        author_list = article.get('AuthorList', [])
        authors = ', '.join([
            f"{a.get('LastName', '')} {a.get('Initials', '')}"
            for a in author_list if 'LastName' in a and 'Initials' in a
        ])
        # Get publication year
        pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
        pub_year = pub_date.get('Year', '')
    # GEO page
    sra_url = f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse_number}'

    # Check for TPM matrix in supplementary files
    tpm_link = ''
    # Fetch GSE record in XML to get supplementary files
    gse_handle = Entrez.efetch(db='gds', id=gse_id, rettype='full', retmode='xml')
    xml_text = gse_handle.read()
    gse_handle.close()
    supp_files = []
    try:
        root = ET.fromstring(xml_text)
        for supp in root.iter('Supplementary-Data'):
            url = supp.text
            if url:
                supp_files.append(url)
        # Find TPM matrix file
        for file_url in supp_files:
            if 'tpm' in file_url.lower() and file_url.lower().endswith(('.txt', '.csv', '.tsv', '.gz')):
                tpm_link = file_url
                break
    except ET.ParseError:
        print(f"Warning: Could not parse XML for GSE {gse_number}. Skipping supplementary file check.")

    results.append({
        'GSE': gse_number,
        'Title': title,
        'Samples': samples,
        'GEO Page': sra_url,
        'PubMed Link': pubmed_link,
        'Authors': authors,
        'Year': pub_year,
        'TPM Matrix': tpm_link
    })

df = pd.DataFrame(results)

# --- Save to file ---
now = datetime.now()
date_str = now.strftime('%Y-%m-%d')
hour_str = now.strftime('%H')
filename = f'geo_search_{first_keyword}_{date_str}_{hour_str}.csv'
df.to_csv(filename, index=False)
print(f"Results saved to {filename}")

# Save to JSON
with open('all_gse_metadata.json', 'w') as f:
    json.dump(all_metadata, f, indent=2)
print("All GSE metadata saved to all_gse_metadata.json")

def get_sra_project_from_gse(gse):
    """Link GSE to SRA project(s) using Entrez elink."""
    handle = Entrez.elink(dbfrom='gds', db='sra', id=gse, linkname='gds_sra')
    record = Entrez.read(handle)
    handle.close()
    sra_projects = []
    try:
        for linksetdb in record[0]['LinkSetDb']:
            for link in linksetdb['Link']:
                sra_projects.append(link['Id'])
    except (KeyError, IndexError):
        pass
    return sra_projects

def get_sra_accession(sra_id):
    """Fetch SRA accession (SRP/SRX/SRR) from SRA UID."""
    handle = Entrez.esummary(db='sra', id=sra_id)
    record = Entrez.read(handle)
    handle.close()
    try:
        exp_xml = record[0]['ExpXml']
        # Find SRP (Study) accession
        import re
        match = re.search(r'<STUDY accession="(SRP[0-9]+)"', exp_xml)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None

def download_run_table(srp, outdir='.'):
    """Download SRA Run Table CSV for a given SRP accession."""
    url = f'https://www.ncbi.nlm.nih.gov/Traces/study/?acc={srp}&go=go'
    # The run selector page has a link to download the run table as CSV
    # But you can also use the direct link:
    csv_url = f'https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi?save=efetch&db=sra&rettype=runinfo&term={srp}'
    response = requests.get(csv_url)
    if response.status_code == 200 and response.text.startswith('Run,'):
        filename = f'{outdir}/{srp}_run_table.csv'
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"Downloaded run table for {srp} to {filename}")
    else:
        print(f"Could not download run table for {srp}")

for gse in df['GSE']:
    print(f"Processing {gse}...")
    sra_ids = get_sra_project_from_gse(gse)
    if not sra_ids:
        print(f"No SRA project found for {gse}")
        continue
    if test_mode:
        sra_ids = sra_ids[:1]
    for sra_id in sra_ids:
        srp = get_sra_accession(sra_id)
        if srp:
            download_run_table(srp)
            time.sleep(1)  # Be polite to NCBI servers
        else:
            print(f"Could not find SRP accession for SRA ID {sra_id}")

print("Done.")


