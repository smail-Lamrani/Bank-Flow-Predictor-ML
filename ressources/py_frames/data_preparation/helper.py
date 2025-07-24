import pandas as pd

import re
import unicodedata
import pandas as pd
import pandas as pd
from tqdm import tqdm

def clean_data(df: pd.DataFrame, callback=None) -> pd.DataFrame:
    """
    Preprocess the DataFrame by:
      1. Renaming columns to more convenient names.
      2. Converting French month names to month numbers.
      3. Combining 'year' and 'month' into a single 'date' column (assuming day=1).
      4. Cleaning all numeric columns by removing unusual whitespace.
      5. Reordering the DataFrame so that 'date' is the first column.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Mapping of original column names to new, convenient names.
    mapping = {
        "Num Tiers": "client",
        "CODE_GUICHET": "code_guichet",
        "Année de DAT_VIREMENT": "year",
        "Mois de DAT_VIREMENT": "month",
        "Nbre virements ADRIA et WEB( BO)": "nb_virements_adria_web_bo",
        "Nbre virements AGENCE, BATCHPERM, FTP et 3 de plus (physique)": "nb_virements_agence_physique",
        "Nbre virements autre digital": "nb_virements_autre_digital",
        "Nbre virements EBICS": "nb_virements_ebics",
        "Nbre virements RPA": "nb_virements_rpa",
        "Mnt Virement ADRIA et WEB( BO)": "mnt_virements_adria_web_bo",
        " Mnt Virement AGENCE, BATCHPERM, FTP et 3 de plus (physique)": "mnt_virements_agence_physique",
        "Mnt Virement autre digital": "mnt_virements_autre_digital",
        "Mnt Virement EBICS": "mnt_virements_ebics",
        "Mnt Virement RPA": "mnt_virements_rpa"
    }
    df = df.rename(columns=mapping)
    
    # Mapping French month names to their numerical equivalents.
    french_months = {
        "janvier": "01",
        "février": "02",
        "mars": "03",
        "avril": "04",
        "mai": "05",
        "juin": "06",
        "juillet": "07",
        "août": "08",
        "septembre": "09",
        "octobre": "10",
        "novembre": "11",
        "décembre": "12"
    }
    
    # Convert the French month names to month numbers.
    df['month_num'] = df['month'].str.strip().str.lower().map(french_months)
    
    # Create the 'date' column by combining 'year' and 'month_num' (day is set to 1).
    df['date'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month_num'] + '-01', 
        errors='coerce'
    )
    
    # Drop the original 'year', 'month', and temporary 'month_num' columns.
    df = df.drop(columns=['year', 'month', 'month_num'])
    
    # Clean all columns (except 'date') by removing any whitespace and converting them to numeric.
    cols = [col for col in df.columns if col not in ('date', 'client', 'code_guichet')]
    total_steps = len(cols)
    for i, col in enumerate(cols):
        df[col] = df[col].astype(str)\
                    .str.replace("\u202F", "", regex=False)\
                    .str.replace("\u00A0", "", regex=False)
        # Convert the cleaned string to a numeric value.
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if callback:
            progress_value = int((i + 1) / total_steps * 100)
            callback(progress_value)
            

    # Reorder the columns so that 'date' is the first column.
    cols = df.columns.tolist()
    cols.remove('date')
    df = df[['date'] + cols]
    
    return df


def get_summary_as_df(df, progress_callback=None):
    if progress_callback:
        progress_callback.omit('Getting summary statistics...')
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df.reset_index(inplace=True)  # Reset index to get column names in the DataFrame
    summary = df.describe(include='all').T  # Transpose for readability

    # Add additional info
    summary['dtype'] = df.dtypes  # Data type of each column
    summary['count'] = df.count()  # Non-null values count
    summary['missing_values'] = df.isnull().sum()  # Count missing values
    summary['missing_percent'] = (df.isnull().sum() / len(df)) * 100  # Percentage of missing values
    summary['unique_values'] = df.nunique()  # Count unique values
    summary['mode'] = df.mode().iloc[0]  # Most frequent value
    summary['skewness'] = df.select_dtypes(include=['number']).skew()  # Skewness for numerical columns
    summary['kurtosis'] = df.select_dtypes(include=['number']).kurtosis()  # Kurtosis for numerical columns
    summary['memory_usage (MB)'] = df.memory_usage(deep=True) / (1024 * 1024)  # Convert to MB

    # Select and reorder columns
    column_order = [
        "Column", "dtype", "count", "missing_values", "missing_percent", 
        "unique_values", "mode", "mean", "std", "min", "max", "skewness", 
        "kurtosis", "memory_usage (MB)"
    ]
    
    # Reset index to move column names into the 'Column' column
    summary = summary.reset_index().rename(columns={'index': 'Column'})
    
    # Keep only existing columns in the specified order
    summary = summary[[col for col in column_order if col in summary.columns]]

    # Format float numbers to three decimal places
    float_cols = summary.select_dtypes(include=['float64', 'float32']).columns
    summary[float_cols] = summary[float_cols].applymap(lambda x: f"{x:.3f}" if pd.notnull(x) else x)
    if progress_callback:
        progress_callback.omit("Summary returned")
    return summary

def split_and_count_by_client(df, years=[], date_col='date',client_col = 'client', check_duplicates_in="M", fill_gaps_to=None, get_num_clients=10):
    # Work on a copy only once.
    df = df.copy()
    if date_col not in df.columns and date_col in df.index.names:
        df.reset_index(date_col, inplace=True)
    
    # Ensure date column is datetime.
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter by years if provided.
    if years:
        df = df[df[date_col].dt.year.isin(years)]
    
    # Validate frequency string.
    valid_freq = ['M', 'Y', 'Q', 'D', 'W', 'H', 'T', 'S']
    if check_duplicates_in not in valid_freq:
        raise ValueError(f"Invalid value for {check_duplicates_in}. Choose from {valid_freq}.")
    
    # Group by client.
    df["client"] = df["client"].astype(str)
    grouped = df.groupby('client')
    clients_info = {}
    clients_dfs = {}
    num_obs_client = {}
    date_min = df[date_col].min()
    date_max = df[date_col].max()

    for client, client_data in tqdm(grouped, desc="Processing clients", unit="client", total=len(grouped)):
        
        # client_data = fill_date_gaps_helper(client_data,
        #                                         client_col = client_col,
        #                                         client_id = str(int(client)),
        #                                         date_col=date_col, 
        #                                         freq=check_duplicates_in,
        #                                         n_obs=fill_gaps_to)
        client_data =  fill_gaps(client_data, date_min, date_max)
        client = str(client)
        clients_dfs[client] = client_data
        num_obs_client[client] = len(client_data)
        # Use vectorized operations rather than per-row loops:
        temp = client_data.copy()
        # Create period column (vectorized).
        temp['dups'] = temp[date_col].dt.to_period(check_duplicates_in)
        # Count duplicates per period.
        temp['dups_count'] = temp.groupby('dups')[date_col].transform('count')
        # Filter periods with duplicates.
        temp = temp[temp['dups_count'] > 1]
        # Aggregate the info per client.
        # Since there's only one client in each group, we simply take the set and the first count.
        agg = temp.groupby('client').agg({'dups': lambda x: set(x), 'dups_count': 'first'}).reset_index()
        clients_info[client] = agg
        if get_num_clients is not None:
            if len(clients_info) >= get_num_clients:
                break

    return clients_dfs, clients_info, num_obs_client

def fill_gaps(df, date_min = None, date_max=None, callback = None):
    if date_min is None or date_max is None:
        date_min = df['date'].min()
        date_max = df['date'].max()
    dfs = []
    df_clients = df.groupby(by='client')
    total_clients = len(df_clients)
    for i, client in enumerate(df_clients):
        client = fill_gaps_(data=client, date_min=date_min, date_max=date_max)
        dfs.append(client)
        if callback:
            progress_value = int(((i + 1) / total_clients) * 100)
            callback(progress_value)
    return pd.concat(dfs)



def fill_gaps_(data, date_min, date_max):
    # 1) Copie de sécurité
    client_id = None
    if isinstance(data, pd.DataFrame):
        df_local = data.copy()
    if isinstance(data, tuple):
        df_local = data[1].copy()
        client_id = data[0]
    else: # Assuming data is a DataFrame if not a tuple
        df_local = data.copy()
    # 2) S'assurer que la colonne date est bien en datetime
    if "date" not in df_local.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'date'.")
    df_local["date"] = pd.to_datetime(df_local["date"])

    # 3) Calcul de la date_min
    if date_min is None:
        date_min = df_local["date"].min()
    if date_max is None:
        date_max = df_local["date"].max()

    date_min = pd.to_datetime(date_min)
    date_max = pd.to_datetime(date_max)

    # 5) Générer la plage de dates mensuelles (du 1er de chaque mois)
    all_months = pd.date_range(start=date_min, end=date_max, freq="MS")

    # 6) Créer un DataFrame indexé par ces mois
    df_all = pd.DataFrame({"date": all_months})

    # 7) Joindre avec df_local (merge ou reindex). On fait un left join
    #    pour garder toutes les lignes de df_all.
    df_merged = pd.merge(df_all, df_local, on="date", how="left")

    # 8) Optionnel : combler les valeurs manquantes
    # (par exemple, 0 transactions, 0 montants, etc.)
    # selon vos colonnes. Ex :
    cols_to_fill = [
        col
        for col in df_merged.columns
        if "nb_virements" in col or "mnt_virements" in col
    ]
    for col in cols_to_fill:
        df_merged[col] = df_merged[col].fillna(0)

    # 9) Tri par date (normalement déjà, mais au cas où)
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    df_merged["client"] = df_local["client"].iloc[0]
    
    if "code_guichet" in df_merged.columns:
        mode_guichet = df_local["code_guichet"].mode()
        if not mode_guichet.empty:
            df_merged["code_guichet"] = mode_guichet[0]

    return df_merged

