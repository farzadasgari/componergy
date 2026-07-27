import pandas as pd
import os
import zipfile
import glob

BASE_DIR = "../dataset/electricity"
PGE_DIR = os.path.join(BASE_DIR, "PGE")
SCE_DIR = os.path.join(BASE_DIR, "SCE")
SDGE_DIR = os.path.join(BASE_DIR, "SDGE")

STD_COLS = [
    'Provider', 'ZipCode', 'Month', 'Year', 'CustomerClass', 
    'Combined', 'TotalCustomers', 'TotalkWh', 'AveragekWh'
]

def clean_numeric_col(series):
    return pd.to_numeric(series.astype(str).str.replace(',', ''), errors='coerce').fillna(0)

def process_pge():
    print("Processing PGE files...")
    df_list = []
    zip_files = glob.glob(os.path.join(PGE_DIR, "*.zip"))
    
    for zf in zip_files:
        try:
            with zipfile.ZipFile(zf, 'r') as z:
                csv_name = [n for n in z.namelist() if n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    temp_df = pd.read_csv(f, usecols=lambda x: x in [
                        'ZipCode', 'Month', 'Year', 'CustomerClass', 
                        'Combined', 'TotalCustomers', 'TotalkWh', 'AveragekWh'
                    ])
                    temp_df['Provider'] = 'PGE'
                    df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {zf}: {e}")

    if not df_list: return pd.DataFrame(columns=STD_COLS)
    return pd.concat(df_list, ignore_index=True)

def process_sce():
    print("Processing SCE files...")
    df_list = []
    excel_files = glob.glob(os.path.join(SCE_DIR, "*.xls*"))
    sce_map = {
        'Zip Code': 'ZipCode',
        'Customer Class': 'CustomerClass',
        'Total Accounts': 'TotalCustomers',
        'Total kWh': 'TotalkWh',
        'Average kWh': 'AveragekWh'
    }
    for xf in excel_files:
        try:
            temp_df = pd.read_excel(xf, dtype={'Zip Code': str})
            temp_df.rename(columns=sce_map, inplace=True)
            cols_to_keep = [c for c in STD_COLS if c in temp_df.columns or c == 'Provider']
            temp_df['Provider'] = 'SCE'
            temp_df = temp_df[cols_to_keep]
            df_list.append(temp_df)
        except Exception as e: print(f"Error reading {xf}: {e}")

    if not df_list: return pd.DataFrame(columns=STD_COLS)
    return pd.concat(df_list, ignore_index=True)

def process_sdge():
    print("Processing SDGE files...")
    df_list = []
    csv_files = glob.glob(os.path.join(SDGE_DIR, "*.csv"))
    
    for cf in csv_files:
        try:
            temp_df = pd.read_csv(cf)
            temp_df.rename(columns={'Zip Code': 'ZipCode', 'Total Accounts': 'TotalCustomers'}, inplace=True)
            cols_in_file = temp_df.columns.tolist()
            valid_cols = [c for c in STD_COLS if c in cols_in_file]
            temp_df = temp_df[valid_cols]
            temp_df['Provider'] = 'SDGE'
            df_list.append(temp_df)
        except Exception as e:  print(f"Error reading {cf}: {e}")

    if not df_list: return pd.DataFrame(columns=STD_COLS)
    return pd.concat(df_list, ignore_index=True)

def main():
    df_pge = process_pge()
    df_sce = process_sce()
    df_sdge = process_sdge()
    print("Merging datasets...")
    full_df = pd.concat([df_pge, df_sce, df_sdge], ignore_index=True)
    print("Cleaning data...")
    for col in STD_COLS:
        if col not in full_df.columns: full_df[col] = None
    full_df = full_df[STD_COLS]
    full_df = full_df.dropna(subset=['ZipCode'])
    full_df = full_df[full_df['ZipCode'] != 0]
    numeric_cols = ['TotalCustomers', 'TotalkWh', 'AveragekWh', 'Month', 'Year']
    for col in numeric_cols:
        full_df[col] = clean_numeric_col(full_df[col])
    full_df['Year'] = full_df['Year'].astype(int)
    full_df['Month'] = full_df['Month'].astype(int)
    output_filename = "../dataset/electricity/electricity_by_county.csv"
    print(f"Saving to {output_filename}...")
    full_df.to_csv(output_filename, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
