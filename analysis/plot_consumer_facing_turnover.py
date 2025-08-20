#!/usr/bin/env python3
"""
Consumer-facing firms turnover distribution plot.
Focuses on sectors most exposed to VAT threshold effects.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_consumer_facing_plot():
    """Create turnover distribution chart for consumer-facing businesses."""
    
    # Load data
    data_path = Path(__file__).parent / 'synthetic_firms_turnover.csv'
    df = pd.read_csv(data_path)
    
    # SIC codes are already numeric in the data
    df['sic_numeric'] = df['sic_code']
    
    # Define consumer-facing sectors most exposed to VAT threshold
    # These are B2C businesses where VAT is a pure cost that can't be reclaimed
    # Using actual SIC codes from the data (these correspond to retail, hospitality, services)
    consumer_facing_sectors = {
        45: 'Motor Vehicle Trade',  # Car sales and repairs
        46: 'Wholesale Trade',  # Some wholesale to small businesses
        47: 'Retail Trade',  # Core retail
        55: 'Accommodation',  # Hotels, B&Bs
        56: 'Food & Beverage Service',  # Restaurants, cafes, pubs
        69: 'Legal & Accounting',  # Professional services to individuals
        70: 'Management Consultancy',  # Business consultancy
        71: 'Architecture & Engineering',  # Services to individuals
        73: 'Advertising',  # Marketing services
        74: 'Other Professional',  # Photography, design, etc.
        77: 'Rental & Leasing',  # Equipment rental
        81: 'Facilities Support',  # Cleaning, landscaping
        85: 'Education',  # Private tutoring, training
        86: 'Human Health',  # Private healthcare
        87: 'Residential Care',  # Care homes
        88: 'Social Work',  # Social services
        90: 'Arts & Entertainment',  # Entertainment venues
        93: 'Sports & Recreation',  # Gyms, sports facilities
        95: 'Repair Services',  # Computer, personal goods repair
        96: 'Other Personal Services'  # Hairdressers, beauty, laundry
    }
    
    # Filter for consumer-facing sectors
    consumer_df = df[df['sic_numeric'].isin(consumer_facing_sectors.keys())]
    
    # Create bins from 1k to 300k with consistent bin sizes
    # Use 1k intervals throughout for consistent visualization
    bin_edges = np.arange(0.5, 300.5, 1.0)  # Consistent 1k intervals
    
    # Calculate weighted histogram
    hist, edges = np.histogram(consumer_df['annual_turnover_k'], bins=bin_edges, weights=consumer_df['weight'])
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Main plot
    x_positions = np.arange(len(hist))
    colors = ['lightcoral' if edges[i] < 90 else 'darkred' if edges[i] < 150 else 'orange' for i in range(len(hist))]
    
    ax1.bar(x_positions, hist, color=colors, alpha=0.7, edgecolor='black', linewidth=0.1)
    
    ax1.set_xlabel('Annual Turnover (£k)')
    ax1.set_ylabel('Number of Consumer-Facing Firms (2024)')
    ax1.set_title('Turnover Distribution - Consumer-Facing Businesses Most Exposed to VAT Threshold')
    
    # Set x-axis labels every 10k
    label_positions = []
    label_texts = []
    for i in range(0, len(edges)-1):
        if edges[i] % 10 == 0:  # Every 10k
            label_positions.append(i)
            label_texts.append(f'{int(edges[i])}')
    ax1.set_xticks(label_positions)
    ax1.set_xticklabels(label_texts, rotation=45)
    
    # Add vertical lines at key thresholds
    threshold_90k = np.where(edges >= 90)[0][0] - 1 if any(edges >= 90) else len(edges)-1
    threshold_150k = np.where(edges >= 150)[0][0] - 1 if any(edges >= 150) else len(edges)-1
    
    ax1.axvline(x=threshold_90k, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax1.axvline(x=threshold_150k, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add labels for the vertical lines
    ax1.text(threshold_90k + 1, max(hist) * 0.9, 'VAT Registration\nThreshold (£90k)', 
             rotation=0, color='red', fontsize=10, ha='left', fontweight='bold')
    ax1.text(threshold_150k + 1, max(hist) * 0.8, 'Flat Rate Scheme\nLimit (£150k)', 
             rotation=0, color='orange', fontsize=10, ha='left')
    
    # Add statistics box
    total_firms = consumer_df['weight'].sum()
    below_threshold = consumer_df[consumer_df['annual_turnover_k'] < 90]['weight'].sum()
    near_threshold = consumer_df[(consumer_df['annual_turnover_k'] >= 80) & 
                                 (consumer_df['annual_turnover_k'] <= 100)]['weight'].sum()
    vat_registered = consumer_df[consumer_df['vat_registered'] == True]['weight'].sum()
    
    stats_text = (f'Consumer-Facing Firms Statistics:\n'
                 f'Total firms: {total_firms:,.0f}\n'
                 f'Below VAT threshold: {below_threshold:,.0f} ({below_threshold/total_firms*100:.1f}%)\n'
                 f'Near threshold (£80-100k): {near_threshold:,.0f} ({near_threshold/total_firms*100:.1f}%)\n'
                 f'VAT registered: {vat_registered:,.0f} ({vat_registered/total_firms*100:.1f}%)')
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Zoomed plot around threshold
    zoom_df = consumer_df[(consumer_df['annual_turnover_k'] >= 70) & 
                          (consumer_df['annual_turnover_k'] <= 110)]
    
    zoom_bins = np.arange(70, 111, 1)  # 1k intervals for zoom
    zoom_hist, zoom_edges = np.histogram(zoom_df['annual_turnover_k'], 
                                         bins=zoom_bins, weights=zoom_df['weight'])
    
    zoom_x = np.arange(len(zoom_hist))
    zoom_colors = ['lightcoral' if zoom_edges[i] < 90 else 'darkred' for i in range(len(zoom_hist))]
    
    ax2.bar(zoom_x, zoom_hist, color=zoom_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Annual Turnover (£k) - Zoomed View')
    ax2.set_ylabel('Number of Firms')
    ax2.set_title('Detail View: Firms Around VAT Threshold (£70k-£110k)')
    
    # Set labels for zoom plot
    zoom_labels = [f'{int(zoom_edges[i])}' for i in range(0, len(zoom_edges)-1, 2)]
    ax2.set_xticks(range(0, len(zoom_hist), 2))
    ax2.set_xticklabels(zoom_labels, rotation=45)
    

    
    # Highlight bunching effect
    bunching_range = zoom_df[(zoom_df['annual_turnover_k'] >= 85) & 
                             (zoom_df['annual_turnover_k'] <= 89)]
    if len(bunching_range) > 0:
        bunching_firms = bunching_range['weight'].sum()
        ax2.text(0.02, 0.98, f'Firms £85-89k: {bunching_firms:,.0f}', 
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save chart
    output_path = Path(__file__).parent / 'turnover_distribution_consumer_facing.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"Chart saved: {output_path}")
    
    # Print sector breakdown
    print("\nConsumer-Facing Sectors Included:")
    print("-" * 50)
    for sector_code, sector_name in consumer_facing_sectors.items():
        sector_firms = df[df['sic_numeric'] == sector_code]['weight'].sum()
        print(f"  {sector_name}: {sector_firms:,.0f} firms")
    
    # Print threshold analysis
    print("\nVAT Threshold Impact Analysis:")
    print("-" * 50)
    
    # Analyze bunching below threshold
    ranges = [
        (0, 50, "£0-50k"),
        (50, 70, "£50-70k"),
        (70, 80, "£70-80k"),
        (80, 85, "£80-85k"),
        (85, 89, "£85-89k (just below threshold)"),
        (90, 95, "£90-95k (just above threshold)"),
        (95, 100, "£95-100k"),
        (100, 150, "£100-150k"),
        (150, 300, "£150-300k")
    ]
    
    print(f"{'Turnover Range':<30} {'Firms':<12} {'% of Total':<10}")
    print("-" * 52)
    
    for min_val, max_val, label in ranges:
        range_firms = consumer_df[(consumer_df['annual_turnover_k'] >= min_val) & 
                                  (consumer_df['annual_turnover_k'] < max_val)]['weight'].sum()
        percentage = (range_firms / total_firms * 100) if total_firms > 0 else 0
        print(f"{label:<30} {range_firms:>10,.0f} {percentage:>8.1f}%")
    
    # Calculate bunching coefficient
    below_threshold_85_89 = consumer_df[(consumer_df['annual_turnover_k'] >= 85) & 
                                        (consumer_df['annual_turnover_k'] < 90)]['weight'].sum()
    above_threshold_90_95 = consumer_df[(consumer_df['annual_turnover_k'] >= 90) & 
                                        (consumer_df['annual_turnover_k'] < 95)]['weight'].sum()
    
    if above_threshold_90_95 > 0:
        bunching_ratio = below_threshold_85_89 / above_threshold_90_95
        print(f"\nBunching Ratio (£85-89k / £90-95k): {bunching_ratio:.2f}")
        print("(Ratio > 1 indicates bunching below threshold)")

if __name__ == "__main__":
    create_consumer_facing_plot()