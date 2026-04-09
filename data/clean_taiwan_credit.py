"""
Download and clean the UCI "Default of Credit Card Clients" dataset.

Source: https://archive.ics.uci.edu/dataset/350/default-of-credit-card-clients
Original paper: Yeh, I-C., & Lien, C-H. (2009). The comparisons of data mining
    techniques for the predictive accuracy of probability of default of credit
    card clients. Expert Systems with Applications, 36(2), 2473-2480.

The original file is an Excel spreadsheet with a double header row and
30,000 records of credit card customers from a bank in Taiwan (April–September 2005).

This script:
  1. Downloads the zip from UCI
  2. Extracts the .xls file
  3. Reads with the descriptive header row (row 1)
  4. Drops the ID column
  5. Renames the target column from "default payment next month" to "default"
  6. Saves as taiwan_credit.csv
"""

import urllib.request
import zipfile
import io
import pandas as pd
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
OUTPUT = Path(__file__).parent / "taiwan_credit.csv"

print("Downloading from UCI...")
resp = urllib.request.urlopen(UCI_URL)
data = resp.read()
print(f"Downloaded {len(data):,} bytes")

z = zipfile.ZipFile(io.BytesIO(data))
xls_name = z.namelist()[0]
print(f"Extracting: {xls_name}")

with z.open(xls_name) as f:
    # Row 0 has generic headers (X1, X2, ..., Y); row 1 has descriptive names
    df = pd.read_excel(f, header=1)

print(f"Raw shape: {df.shape}")

# Drop the ID column (just a row number, not a feature)
df = df.drop(columns=["ID"])

# Rename target for brevity
df = df.rename(columns={"default payment next month": "default"})

print(f"Clean shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nDefault rate: {df['default'].mean():.3f} ({df['default'].sum():,} / {len(df):,})")

df.to_csv(OUTPUT, index=False)
print(f"\nSaved to {OUTPUT}")
