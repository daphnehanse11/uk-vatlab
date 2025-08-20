#!/usr/bin/env python3
"""
Sector-specific turnover distribution plots from synthetic firm data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def create_sector_turnover_plot(sector_code=None, save_all=False):
    """Create sector-specific turnover distribution charts.
    
    Args:
        sector_code: Specific SIC code to plot (e.g., '1' for sector 1)
                    If None, will create plots for all sectors
        save_all: If True, save individual plots for all sectors
    """
    
    # Load data
    data_path = Path(__file__).parent / 'synthetic_firms_turnover.csv'
    df = pd.read_csv(data_path)
    
    # Convert SIC code to numeric for easier filtering
    df['sic_numeric'] = df['sic_code'].astype(str).str[:1].astype(int)
    
    # SIC code to sector name mapping
    sector_names = {
        1: 'Agriculture, Forestry & Fishing',
        2: 'Mining & Quarrying',
        3: 'Manufacturing',
        4: 'Electricity, Gas, Steam & Air',
        5: 'Water Supply & Waste',
        6: 'Construction',
        7: 'Wholesale & Retail Trade',
        8: 'Transportation & Storage',
        9: 'Accommodation & Food Service',
        10: 'Information & Communication',
        11: 'Financial & Insurance',
        12: 'Real Estate',
        13: 'Professional & Technical',
        14: 'Administrative & Support',
        15: 'Public Admin & Defence',
        16: 'Education',
        17: 'Health & Social Work',
        18: 'Arts & Entertainment',
        19: 'Other Service Activities'
    }
    
    if sector_code is not None:
        # Plot single sector
        plot_single_sector(df, int(sector_code), sector_names.get(int(sector_code), f'Sector {sector_code}'))
    elif save_all:
        # Create plots for all sectors
        unique_sectors = sorted(df['sic_numeric'].unique())
        for sector in unique_sectors:
            sector_name = sector_names.get(sector, f'Sector {sector}')
            plot_single_sector(df, sector, sector_name, save=True)
            print(f"Created plot for {sector_name}")
    else:
        # Create a comparison plot showing top sectors
        plot_sector_comparison(df, sector_names)

def plot_single_sector(df, sector_code, sector_name, save=True):
    """Plot turnover distribution for a single sector."""
    
    # Filter for specific sector
    sector_df = df[df['sic_numeric'] == sector_code]
    
    if len(sector_df) == 0:
        print(f"No data found for sector {sector_code}")
        return
    
    # Create bins from 1k to 300k
    bin_edges = np.arange(0.5, 300.5, 1.0)  # 1k intervals
    
    # Calculate weighted histogram
    hist, _ = np.histogram(sector_df['annual_turnover_k'], bins=bin_edges, weights=sector_df['weight'])
    
    # Create plot
    plt.figure(figsize=(15, 6))
    x_positions = np.arange(len(hist))
    
    plt.bar(x_positions, hist, color='lightcoral', alpha=0.7, edgecolor='black', linewidth=0.1)
    
    plt.xlabel('Annual Turnover (£k)')
    plt.ylabel(f'Number of Firms in {sector_name} (2024)')
    plt.title(f'Turnover Distribution - {sector_name}')
    
    # Set x-axis labels every 10k
    label_positions = [i for i in range(9, len(hist), 10)]  # Start at 10k (index 9)
    label_texts = [f'{i+1}' for i in label_positions]
    plt.xticks(label_positions, label_texts)
    
    # Add vertical lines at 90k and 150k
    plt.axvline(x=89, color='red', linestyle='--', alpha=0.7, linewidth=2)  # 90k threshold
    plt.axvline(x=149, color='red', linestyle='--', alpha=0.7, linewidth=2)  # 150k threshold
    
    # Add labels for the vertical lines
    plt.text(90, max(hist) * 0.8, 'VAT Threshold', rotation=0, color='red', fontsize=8, ha='left')
    plt.text(150, max(hist) * 0.8, 'VAT Flat Rate Scheme', rotation=0, color='red', fontsize=8, ha='left')
    
    # Add statistics
    total_firms = sector_df['weight'].sum()
    median_turnover = np.median(sector_df['annual_turnover_k'])
    vat_registered = sector_df[sector_df['vat_registered'] == True]['weight'].sum()
    vat_percentage = (vat_registered / total_firms * 100) if total_firms > 0 else 0
    
    stats_text = f'Total firms: {total_firms:,.0f}\nMedian turnover: £{median_turnover:,.0f}k\nVAT registered: {vat_percentage:.1f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    if save:
        # Save chart
        output_path = Path(__file__).parent / f'turnover_distribution_sector_{sector_code}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_sector_comparison(df, sector_names):
    """Create a comparison plot showing multiple sectors."""
    
    # Get top 5 sectors by firm count
    sector_counts = df.groupby('sic_numeric')['weight'].sum().sort_values(ascending=False).head(5)
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Create bins
    bin_edges = np.arange(0.5, 300.5, 1.0)
    
    for idx, (sector_code, _) in enumerate(sector_counts.items()):
        if idx >= 6:
            break
            
        ax = axes[idx] if idx < 5 else axes[5]
        sector_df = df[df['sic_numeric'] == sector_code]
        sector_name = sector_names.get(sector_code, f'Sector {sector_code}')
        
        # Calculate weighted histogram
        hist, _ = np.histogram(sector_df['annual_turnover_k'], bins=bin_edges, weights=sector_df['weight'])
        
        # Plot
        x_positions = np.arange(len(hist))
        ax.bar(x_positions, hist, color=f'C{idx}', alpha=0.7, edgecolor='black', linewidth=0.1)
        
        ax.set_xlabel('Annual Turnover (£k)')
        ax.set_ylabel('Number of Firms')
        ax.set_title(f'{sector_name}', fontsize=10)
        
        # Set x-axis labels every 50k
        label_positions = [i for i in range(49, len(hist), 50)]
        label_texts = [f'{i+1}' for i in label_positions]
        ax.set_xticks(label_positions)
        ax.set_xticklabels(label_texts)
        
        # Add VAT threshold lines
        ax.axvline(x=89, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=149, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide the 6th subplot if we have only 5 sectors
    if idx < 5:
        axes[5].set_visible(False)
    
    plt.suptitle('Turnover Distribution by Top Sectors (2024)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save chart
    output_path = Path(__file__).parent / 'turnover_distribution_sectors_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Comparison chart saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate sector-specific turnover distribution plots')
    parser.add_argument('--sector', type=str, help='Specific sector code (1-19) to plot')
    parser.add_argument('--all', action='store_true', help='Generate plots for all sectors')
    parser.add_argument('--comparison', action='store_true', help='Generate comparison plot (default)')
    
    args = parser.parse_args()
    
    if args.sector:
        create_sector_turnover_plot(sector_code=args.sector)
    elif args.all:
        create_sector_turnover_plot(save_all=True)
    else:
        create_sector_turnover_plot()  # Default: comparison plot

if __name__ == "__main__":
    main()