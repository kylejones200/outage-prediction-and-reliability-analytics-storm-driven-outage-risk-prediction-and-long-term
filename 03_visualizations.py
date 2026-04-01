import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Visualization script for Outage Detection and Analysis Blog
Generates publication-quality figures at 300 DPI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from datetime import datetime, timedelta

import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS






def generate_outage_architecture():
    """Generate outage analysis system architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis('off')
    
    # EAGLE-I data source
    y_source = 8
    source_box = FancyBboxPatch((1.5, y_source), 3, 1.2, 
                               boxstyle="round,pad=0.1", 
                               facecolor=COLORS['black'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(source_box)
    ax.text(3, y_source + 0.6, 'EAGLE-I Dataset\nOak Ridge National Lab\n15-min Updates', 
           ha='center', va='center', fontsize=10, color='white', weight='bold')
    
    # Data processing layer
    y_processing = 6.5
    processing_items = [
        {'name': 'Parquet\nArchives', 'x': 0.5, 'color': COLORS['darkgray']},
        {'name': 'Real-Time\nIngestion', 'x': 2.5, 'color': COLORS['darkgray']},
        {'name': 'County\nMapping', 'x': 4.5, 'color': COLORS['darkgray']}
    ]
    
    for item in processing_items:
        box = FancyBboxPatch((item['x'], y_processing), 1.5, 0.8, 
                            boxstyle="round,pad=0.05",
                            facecolor=item['color'], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(item['x'] + 0.75, y_processing + 0.4, item['name'], 
               ha='center', va='center', fontsize=8, color='white', weight='bold')
    
    # Service layer
    y_service = 5
    service_box = FancyBboxPatch((0.5, y_service), 5.5, 1, 
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['gray'], 
                                edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(service_box)
    ax.text(3.25, y_service + 0.5, 'EagleiOutageService\nQuery • Filter • Temporal Analysis • Aggregation', 
           ha='center', va='center', fontsize=10, color='white', weight='bold')
    
    # Analysis modules
    y_analysis = 3.5
    modules = [
        {'name': 'Event\nDetection', 'x': 0.5, 'color': COLORS['darkgray']},
        {'name': 'Severity\nClassification', 'x': 2, 'color': COLORS['darkgray']},
        {'name': 'Temporal\nPatterns', 'x': 3.5, 'color': COLORS['darkgray']},
        {'name': 'Load\nImpact', 'x': 5, 'color': COLORS['darkgray']}
    ]
    
    for module in modules:
        box = FancyBboxPatch((module['x'], y_analysis), 1.3, 0.8, 
                            boxstyle="round,pad=0.05",
                            facecolor=module['color'], 
                            edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(module['x'] + 0.65, y_analysis + 0.4, module['name'], 
               ha='center', va='center', fontsize=8, color='white', weight='bold')
    
    # Alert and response layer
    y_alert = 2
    alert_box = FancyBboxPatch((1, y_alert), 4.5, 1, 
                              boxstyle="round,pad=0.1",
                              facecolor=COLORS['gray'], 
                              edgecolor='black', linewidth=2, alpha=0.9)
    ax.add_patch(alert_box)
    ax.text(3.25, y_alert + 0.5, 'Alert & Response System\nThreshold Monitoring • Notifications • Dashboards', 
           ha='center', va='center', fontsize=10, color='white', weight='bold')
    
    # Output layer
    y_output = 0.3
    outputs = [
        {'name': 'Operator\nAlerts', 'x': 0.5},
        {'name': 'Load\nAdjustment', 'x': 2},
        {'name': 'Web\nDashboard', 'x': 3.5},
        {'name': 'API\nEndpoints', 'x': 5}
    ]
    
    for output in outputs:
        box = FancyBboxPatch((output['x'], y_output), 1.3, 0.6, 
                            boxstyle="round,pad=0.05",
                            facecolor=COLORS['black'], 
                            edgecolor='black', linewidth=1.5, alpha=0.7)
        ax.add_patch(box)
        ax.text(output['x'] + 0.65, y_output + 0.3, output['name'], 
               ha='center', va='center', fontsize=8, color='white', weight='bold')
    
    # Integration box (right side)
    y_integration = 2
    integration_box = FancyBboxPatch((7, y_integration), 3, 6, 
                                    boxstyle="round,pad=0.15",
                                    facecolor=COLORS['gray'], 
                                    edgecolor='black', linewidth=2, alpha=0.3)
    ax.add_patch(integration_box)
    ax.text(8.5, 7.5, 'System Integration', ha='center', fontsize=11, weight='bold')
    
    integration_items = [
        {'name': 'Load Forecasting', 'y': 6.8},
        {'name': 'Weather Data', 'y': 6.1},
        {'name': 'Transmission Map', 'y': 5.4},
        {'name': 'Emergency\nResponse', 'y': 4.7},
        {'name': 'Restoration\nTracking', 'y': 4},
        {'name': 'Reporting', 'y': 3.3}
    ]
    
    for item in integration_items:
        box = FancyBboxPatch((7.3, item['y']), 2.4, 0.5, 
                            boxstyle="round,pad=0.05",
                            facecolor='white', 
                            edgecolor=COLORS['gray'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(8.5, item['y'] + 0.25, item['name'], 
               ha='center', va='center', fontsize=8)
    
    # Draw arrows
    ax.arrow(3, y_source, 0, -1.3, head_width=0.2, head_length=0.15, 
            fc='black', ec='black', linewidth=2)
    ax.arrow(3.25, y_service, 0, -1.3, head_width=0.2, head_length=0.15, 
            fc='black', ec='black', linewidth=2)
    ax.arrow(3.25, y_alert, 0, -1.5, head_width=0.2, head_length=0.15, 
            fc='black', ec='black', linewidth=2)
    
    # Integration arrows
    for y_pos in [6.8, 5.4, 4]:
        ax.arrow(6.9, y_pos + 0.25, -0.5, 0, head_width=0.15, head_length=0.1, 
                fc=COLORS['gray'], ec=COLORS['gray'], linewidth=1.5, alpha=0.6)
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 9.5)
    
    ax.text(5.5, 9.3, 'Outage Detection & Analysis Architecture', 
           ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    plt.savefig('03_outage_analysis_architecture.png', bbox_inches='tight', dpi=300)
    print("✓ Generated: 03_outage_analysis_architecture.png")
    plt.close()


def generate_outage_dashboard():
    """Generate comprehensive outage analysis dashboard."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # Major outage events timeline
    ax1 = fig.add_subplot(gs[0, :])
    dates = pd.date_range('2024-01-01', periods=12, freq='MS')
    events_critical = [1, 0, 2, 1, 3, 2, 4, 5, 3, 2, 1, 1]
    events_major = [3, 2, 4, 3, 6, 5, 8, 10, 7, 4, 3, 2]
    events_significant = [12, 10, 15, 14, 18, 20, 25, 28, 22, 18, 15, 12]
    
    ax1.bar(dates, events_critical, width=20, label='Critical (500k+ customers)', 
           color=COLORS['black'], edgecolor='black', linewidth=1)
    ax1.bar(dates, events_major, width=20, bottom=events_critical, 
           label='Major (100k-500k)', color=COLORS['darkgray'], edgecolor='black', linewidth=1)
    ax1.bar(dates, events_significant, width=20, 
           bottom=np.array(events_critical)+np.array(events_major),
           label='Significant (10k-100k)', color=COLORS['gray'], edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Month (2024)', fontsize=11, weight='bold')
    ax1.set_ylabel('Number of Events', fontsize=11, weight='bold')
    ax1.set_title('Major Outage Events by Severity (2024)', fontsize=12, weight='bold')
    ax1.legend(loc='upper right', ncol=3)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # State outage summary
    ax2 = fig.add_subplot(gs[1, :2])
    states = ['Texas', 'Florida', 'California', 'Louisiana', 'Michigan', 
              'New York', 'Pennsylvania', 'Ohio']
    total_customers_out = [5200000, 4800000, 3200000, 2800000, 2400000, 
                          2200000, 1900000, 1700000]
    avg_outage_rate = [3.2, 4.8, 2.1, 5.2, 3.8, 2.9, 2.4, 2.6]
    
    x = np.arange(len(states))
    width = 0.35
    
    ax2_1 = ax2
    bars1 = ax2_1.bar(x - width/2, [c/1000000 for c in total_customers_out], width, 
                     label='Total Customers Out (Millions)', 
                     color=COLORS['black'], edgecolor='black', linewidth=1)
    
    ax2_2 = ax2.twinx()
    bars2 = ax2_2.bar(x + width/2, avg_outage_rate, width, 
                     label='Avg Outage Rate (%)', 
                     color=COLORS['gray'], edgecolor='black', linewidth=1)
    
    ax2_1.set_xlabel('State', fontsize=11, weight='bold')
    ax2_1.set_ylabel('Total Customers Out (Millions)', fontsize=10, color=COLORS['black'])
    ax2_2.set_ylabel('Avg Outage Rate (%)', fontsize=10, color=COLORS['gray'])
    ax2_1.set_title('State Outage Summary (2024)', fontsize=12, weight='bold')
    ax2_1.set_xticks(x)
    ax2_1.set_xticklabels(states, rotation=45, ha='right')
    ax2_1.tick_params(axis='y', labelcolor=COLORS['black'])
    ax2_2.tick_params(axis='y', labelcolor=COLORS['gray'])
    ax2_1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Legend combining both axes
    lines1, labels1 = ax2_1.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Severity distribution pie
    ax3 = fig.add_subplot(gs[1, 2])
    severity_labels = ['Critical', 'Major', 'Significant', 'Minor']
    severity_counts = [28, 67, 209, 896]
    colors_severity = [COLORS['black'], COLORS['darkgray'], COLORS['gray'], COLORS['lightgray']]
    
    wedges, texts, autotexts = ax3.pie(severity_counts, labels=severity_labels, 
                                       autopct='%1.1f%%', colors=colors_severity,
                                       startangle=90, textprops={'weight': 'bold', 'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
    ax3.set_title('Event Severity Distribution', fontsize=11, weight='bold')
    
    # Hourly outage pattern
    ax4 = fig.add_subplot(gs[2, 0])
    hours = np.arange(24)
    avg_customers_out = [
        1200, 1100, 1000, 950, 900, 950, 1100, 1300,
        1500, 1600, 1700, 1800, 1900, 2100, 2400, 2800,
        3200, 3500, 3200, 2800, 2200, 1900, 1600, 1400
    ]
    
    ax4.plot(hours, avg_customers_out, linewidth=2.5, color=COLORS['black'], 
            marker='o', markersize=4, label='Avg Customers Out')
    ax4.fill_between(hours, avg_customers_out, alpha=0.3, color=COLORS['black'])
    ax4.axvline(x=17, color=COLORS['black'], linestyle='--', linewidth=2, 
                label='Peak Hour (17:00)')
    ax4.set_xlabel('Hour of Day', fontsize=10, weight='bold')
    ax4.set_ylabel('Avg Customers Out (1000s)', fontsize=10, weight='bold')
    ax4.set_title('Hourly Outage Pattern', fontsize=11, weight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xticks(hours[::3])
    
    # Monthly pattern
    ax5 = fig.add_subplot(gs[2, 1])
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    total_events = [125, 98, 142, 156, 188, 205, 268, 295, 234, 186, 145, 128]
    colors_months = [COLORS['gray'] if e < 200 else COLORS['gray'] 
                     if e < 250 else COLORS['black'] for e in total_events]
    
    bars = ax5.bar(range(12), total_events, color=colors_months, 
                   edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Total Events', fontsize=10, weight='bold')
    ax5.set_title('Monthly Event Count', fontsize=11, weight='bold')
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(months, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax5.axhline(y=200, color=COLORS['black'], linestyle='--', linewidth=1.5, 
                alpha=0.7, label='High Activity Threshold')
    ax5.legend(fontsize=8)
    
    # Load impact analysis
    ax6 = fig.add_subplot(gs[2, 2])
    time_points = np.arange(24)
    forecast_load = 25000 + 3000 * np.sin(time_points * np.pi / 12)
    observed_load = forecast_load.copy()
    # Simulate outage event from hour 10-16
    observed_load[10:16] *= 0.75  # 25% load drop during outage
    
    ax6.plot(time_points, forecast_load, linewidth=2, color=COLORS['black'], 
            label='Forecast Load', linestyle='--')
    ax6.plot(time_points, observed_load, linewidth=2, color=COLORS['black'], 
            label='Observed Load')
    ax6.axvspan(10, 16, alpha=0.3, color=COLORS['black'], label='Outage Period')
    ax6.set_xlabel('Hour', fontsize=10, weight='bold')
    ax6.set_ylabel('Load (MW)', fontsize=10, weight='bold')
    ax6.set_title('Load Impact Analysis', fontsize=11, weight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    # Add annotation
    ax6.annotate('Outage-Driven\nLoad Drop', xy=(13, observed_load[13]), 
                xytext=(15, 22000),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=9, weight='bold')
    
    plt.suptitle('Outage Detection & Analysis Dashboard', 
                fontsize=14, weight='bold', y=0.995)
    
    plt.savefig('03_outage_analysis_dashboard.png', bbox_inches='tight', dpi=300)
    print("✓ Generated: 03_outage_analysis_dashboard.png")
    plt.close()


def generate_temporal_patterns():
    """Generate detailed temporal pattern analysis."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Hourly pattern with confidence intervals
    ax1 = fig.add_subplot(gs[0, :])
    hours = np.arange(24)
    avg_outages = 1200 + 800 * np.sin((hours - 8) * np.pi / 12) ** 2
    upper_bound = avg_outages * 1.5
    lower_bound = avg_outages * 0.5
    
    ax1.plot(hours, avg_outages, linewidth=3, color=COLORS['black'], 
            label='Average Customers Out', marker='o', markersize=6)
    ax1.fill_between(hours, lower_bound, upper_bound, alpha=0.2, 
                    color=COLORS['black'], label='Confidence Interval')
    
    # Highlight peak hours
    peak_start = 14
    peak_end = 19
    ax1.axvspan(peak_start, peak_end, alpha=0.2, color=COLORS['black'])
    ax1.text(16.5, max(upper_bound) * 0.9, 'Peak\nVulnerability\nWindow', 
            ha='center', fontsize=10, weight='bold', color=COLORS['black'])
    
    ax1.set_xlabel('Hour of Day', fontsize=11, weight='bold')
    ax1.set_ylabel('Average Customers Out (1000s)', fontsize=11, weight='bold')
    ax1.set_title('Hourly Outage Pattern with Confidence Intervals', 
                 fontsize=12, weight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(hours[::2])
    
    # Day of week pattern
    ax2 = fig.add_subplot(gs[1, 0])
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_events = [145, 152, 148, 156, 168, 195, 186]
    dow_colors = [COLORS['darkgray'] if e < 160 else COLORS['gray'] 
                  if e < 180 else COLORS['black'] for e in dow_events]
    
    bars = ax2.bar(range(7), dow_events, color=dow_colors, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Number of Events', fontsize=10, weight='bold')
    ax2.set_title('Outage Events by Day of Week', fontsize=11, weight='bold')
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(dow_labels)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for i, (bar, count) in enumerate(zip(bars, dow_events)):
        ax2.text(i, bar.get_height() + 3, str(count), 
                ha='center', va='bottom', fontsize=9, weight='bold')
    
    # Seasonal pattern
    ax3 = fig.add_subplot(gs[1, 1])
    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    customer_hours = [2.8, 2.2, 3.1, 3.4, 4.2, 4.8, 6.2, 6.8, 5.4, 4.1, 3.2, 2.9]
    
    ax3.plot(range(12), customer_hours, linewidth=3, color=COLORS['black'], 
            marker='s', markersize=8, label='Customer-Outage-Hours (Millions)')
    ax3.fill_between(range(12), customer_hours, alpha=0.3, color=COLORS['black'])
    
    # Highlight summer peak
    ax3.axvspan(5, 8, alpha=0.2, color=COLORS['gray'])
    ax3.text(6.5, 6, 'Summer\nPeak', ha='center', fontsize=9, 
            weight='bold', color=COLORS['black'])
    
    ax3.set_xlabel('Month', fontsize=10, weight='bold')
    ax3.set_ylabel('Customer-Outage-Hours (Millions)', fontsize=10, weight='bold')
    ax3.set_title('Seasonal Outage Pattern', fontsize=11, weight='bold')
    ax3.set_xticks(range(12))
    ax3.set_xticklabels(months)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Temporal Outage Pattern Analysis', 
                fontsize=14, weight='bold')
    
    plt.savefig('03_outage_temporal_patterns.png', bbox_inches='tight', dpi=300)
    print("✓ Generated: 03_outage_temporal_patterns.png")
    plt.close()


if __name__ == "__main__":
    print("Generating visualizations for Outage Detection & Analysis Blog...\n")
    
    generate_outage_architecture()
    generate_outage_dashboard()
    generate_temporal_patterns()
    
    print("\n✓ All visualizations generated successfully!")
    print("  - 03_outage_analysis_architecture.png")
    print("  - 03_outage_analysis_dashboard.png")
    print("  - 03_outage_temporal_patterns.png")

