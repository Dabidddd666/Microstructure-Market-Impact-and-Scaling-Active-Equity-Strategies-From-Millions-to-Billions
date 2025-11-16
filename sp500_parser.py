#!/usr/bin/env python3
"""
S&P 500 Companies Data Parser
Extracts ticker symbols, GICS sectors, and sub-industries from Wikipedia data
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import csv

def parse_sp500_data():
    """
    Parse S&P 500 data from Wikipedia and extract ticker, sector, and sub-industry information
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        # Get the webpage
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main table
        table = soup.find('table', {'class': 'wikitable sortable'})
        
        if not table:
            print("Could not find the S&P 500 table")
            return None
        
        # Extract data from table
        data = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 6:  # Ensure we have enough columns
                ticker = cells[0].get_text(strip=True)
                security = cells[1].get_text(strip=True)
                gics_sector = cells[2].get_text(strip=True)
                gics_sub_industry = cells[3].get_text(strip=True)
                headquarters = cells[4].get_text(strip=True)
                date_added = cells[5].get_text(strip=True)
                
                # Only add if we have a valid ticker
                if ticker and ticker != '':
                    data.append({
                        'Ticker': ticker,
                        'Security': security,
                        'GICS_Sector': gics_sector,
                        'GICS_Sub_Industry': gics_sub_industry,
                        'Headquarters': headquarters,
                        'Date_Added': date_added
                    })
        
        return data
    
    except Exception as e:
        print(f"Error parsing data: {e}")
        return None

def save_to_csv(data, filename):
    """Save data to CSV file"""
    if not data:
        print("No data to save")
        return
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    print(f"Total companies: {len(data)}")

def save_to_json(data, filename):
    """Save data to JSON file"""
    if not data:
        print("No data to save")
        return
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {filename}")

def print_summary(data):
    """Print summary statistics"""
    if not data:
        print("No data to summarize")
        return
    
    df = pd.DataFrame(data)
    
    print("\n=== S&P 500 Companies Summary ===")
    print(f"Total companies: {len(data)}")
    
    print("\n=== GICS Sectors ===")
    sector_counts = df['GICS_Sector'].value_counts()
    for sector, count in sector_counts.items():
        print(f"{sector}: {count}")
    
    print("\n=== Sample Data (First 10 companies) ===")
    print(df[['Ticker', 'Security', 'GICS_Sector', 'GICS_Sub_Industry']].head(10).to_string(index=False))

