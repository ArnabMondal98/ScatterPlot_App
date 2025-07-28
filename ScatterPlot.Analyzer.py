"""
VS Code Excel Scatter Plot App with Z-Score Transformation
=========================================================

Interactive scatter plot analyzer designed for VS Code environment.
Features local file picker, Z-score transformations, size mapping, and comprehensive statistics.

Usage:
1. Run in VS Code with Python extension
2. Use interactive file picker to select Excel files
3. Choose variables, transformations, and size mapping
4. Generate plots and statistical analysis

Requirements: pip install pandas plotly numpy scipy openpyxl
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExcelScatterAnalyzer:
    """Excel Scatter Plot Analyzer with Z-Score Transformation and Size Mapping"""
    
    def __init__(self):
        self.df = None
        self.file_path = None
        self.analysis_results = {}
        
    def list_excel_files(self, directory=None):
        """List all Excel files in specified directory or current directory"""
        if directory is None:
            directory = os.getcwd()
        
        excel_patterns = ['*.xlsx', '*.xls']
        excel_files = []
        
        for pattern in excel_patterns:
            excel_files.extend(glob.glob(os.path.join(directory, pattern)))
            # Also check subdirectories
            excel_files.extend(glob.glob(os.path.join(directory, '**', pattern), recursive=True))
        
        return sorted(excel_files)
    
    def interactive_file_picker(self):
        """Interactive file picker for VS Code"""
        print("🔍 EXCEL FILE PICKER")
        print("=" * 50)
        
        # Option 1: Enter file path directly
        print("Option 1: Enter file path directly")
        file_path = input("Enter Excel file path (or press Enter to browse): ").strip()
        
        if file_path and os.path.exists(file_path):
            return file_path
        
        # Option 2: Browse current directory
        print("\nOption 2: Select from available Excel files")
        current_dir = os.getcwd()
        excel_files = self.list_excel_files(current_dir)
        
        if not excel_files:
            print("❌ No Excel files found in current directory")
            print(f"Current directory: {current_dir}")
            
            # Option 3: Browse different directory
            new_dir = input("Enter directory path to search (or press Enter to skip): ").strip()
            if new_dir and os.path.exists(new_dir):
                excel_files = self.list_excel_files(new_dir)
            
            if not excel_files:
                print("❌ No Excel files found")
                return None
        
        print(f"\n📁 Found {len(excel_files)} Excel files:")
        for i, file_path in enumerate(excel_files, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"{i:2d}. {file_name} ({file_size:.1f} KB)")
        
        try:
            choice = int(input(f"\nSelect file (1-{len(excel_files)}): "))
            if 1 <= choice <= len(excel_files):
                return excel_files[choice - 1]
            else:
                print("❌ Invalid selection")
                return None
        except ValueError:
            print("❌ Invalid input")
            return None
    
    def load_excel(self, file_path=None):
        """Load Excel file with error handling"""
        if file_path is None:
            file_path = self.interactive_file_picker()
        
        if file_path is None:
            return False
        
        try:
            print(f"\n📊 Loading: {os.path.basename(file_path)}")
            self.df = pd.read_excel(file_path)
            self.file_path = file_path
            
            print(f"✅ Successfully loaded!")
            print(f"📋 Dataset shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            print(f"🔢 Numeric columns: {len(self.df.select_dtypes(include=[np.number]).columns)}")
            
            # Show column info
            print("\n📊 Column Information:")
            for i, col in enumerate(self.df.columns, 1):
                dtype = str(self.df[col].dtype)
                null_count = self.df[col].isnull().sum()
                print(f"{i:2d}. {col:<20} ({dtype:<10}) - {null_count} nulls")
            
            # Show data preview
            print("\n📋 Data Preview:")
            print(self.df.head(3).to_string(max_cols=6))
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading file: {str(e)}")
            return False
    
    def select_variables(self):
        """Interactive variable selection including size variable"""
        if self.df is None:
            print("❌ No data loaded. Please load Excel file first.")
            return None, None, None, None
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = self.df.columns.tolist()
        
        if len(numeric_cols) < 2:
            print("❌ Need at least 2 numeric columns for scatter plot")
            return None, None, None, None
        
        print("\n🎯 VARIABLE SELECTION")
        print("=" * 50)
        
        # X variable selection
        print("📊 Select X-axis variable:")
        for i, col in enumerate(numeric_cols, 1):
            print(f"{i:2d}. {col}")
        
        try:
            x_choice = int(input(f"Enter choice (1-{len(numeric_cols)}): "))
            if 1 <= x_choice <= len(numeric_cols):
                x_var = numeric_cols[x_choice - 1]
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        # Y variable selection
        print(f"\n📈 Select Y-axis variable (X-axis: {x_var}):")
        for i, col in enumerate(numeric_cols, 1):
            marker = " ← X-axis" if col == x_var else ""
            print(f"{i:2d}. {col}{marker}")
        
        try:
            y_choice = int(input(f"Enter choice (1-{len(numeric_cols)}): "))
            if 1 <= y_choice <= len(numeric_cols):
                y_var = numeric_cols[y_choice - 1]
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        # Color variable selection (optional)
        print(f"\n🎨 Select color variable (optional):")
        print("0. None")
        for i, col in enumerate(all_cols, 1):
            print(f"{i:2d}. {col}")
        
        try:
            color_choice = int(input(f"Enter choice (0-{len(all_cols)}): "))
            if color_choice == 0:
                color_var = None
            elif 1 <= color_choice <= len(all_cols):
                color_var = all_cols[color_choice - 1]
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        # Size variable selection (optional)
        print(f"\n📏 Select size variable (optional, numeric only):")
        print("0. None")
        for i, col in enumerate(numeric_cols, 1):
            print(f"{i:2d}. {col}")
        
        try:
            size_choice = int(input(f"Enter choice (0-{len(numeric_cols)}): "))
            if size_choice == 0:
                size_var = None
            elif 1 <= size_choice <= len(numeric_cols):
                size_var = numeric_cols[size_choice - 1]
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        return x_var, y_var, color_var, size_var
    
    def select_transformations(self):
        """Select Z-score transformation options"""
        print("\n🔄 Z-SCORE TRANSFORMATION OPTIONS")
        print("=" * 50)
        print("Z-score formula: Z = (X - μ) / σ")
        print("Benefits: Standardizes different scales, identifies outliers")
        
        # X transformation
        x_transform = input(f"Apply Z-score to X-axis? (y/N): ").lower().startswith('y')
        
        # Y transformation
        y_transform = input(f"Apply Z-score to Y-axis? (y/N): ").lower().startswith('y')
        
        # Outlier highlighting
        show_outliers = input(f"Highlight outliers (|Z| > 2)? (Y/n): ").lower() != 'n'
        
        # Trend line
        show_trend = input(f"Show trend line? (Y/n): ").lower() != 'n'
        
        return x_transform, y_transform, show_outliers, show_trend
    
    def calculate_z_score(self, data):
        """Calculate Z-score: Z = (X - μ) / σ"""
        return (data - np.mean(data)) / np.std(data, ddof=1)
    
    def calculate_statistics(self, x_data, y_data):
        """Calculate comprehensive statistics"""
        # Remove infinite and NaN values
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) < 2:
            return None
        
        # Basic statistics
        correlation = np.corrcoef(x_clean, y_clean)[0, 1]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        # Additional statistics
        x_mean, x_std = np.mean(x_clean), np.std(x_clean, ddof=1)
        y_mean, y_std = np.mean(y_clean), np.std(y_clean, ddof=1)
        
        return {
            'correlation': correlation,
            'r_squared': r_value**2,
            'slope': slope,
            'intercept': intercept,
            'p_value': p_value,
            'std_error': std_err,
            'n_points': len(x_clean),
            'x_mean': x_mean,
            'x_std': x_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
    
    def create_scatter_plot(self, x_var, y_var, color_var=None, size_var=None,
                           x_transform=False, y_transform=False, 
                           show_outliers=True, show_trend=True):
        """Create interactive scatter plot with Plotly, including size mapping"""
        
        if self.df is None:
            print("❌ No data loaded")
            return None
        
        print(f"\n🎨 Creating scatter plot: {y_var} vs {x_var}")
        
        # Prepare data
        columns = [x_var, y_var]
        if color_var:
            columns.append(color_var)
        if size_var:
            columns.append(size_var)
        plot_df = self.df[columns].dropna()
        
        if len(plot_df) == 0:
            print("❌ No valid data points after removing missing values")
            return None
        
        # Get data arrays
        x_data = plot_df[x_var].astype(float).values
        y_data = plot_df[y_var].astype(float).values
        
        # Store original data for outlier detection
        x_original = x_data.copy()
        y_original = y_data.copy()
        
        # Apply transformations
        x_label = x_var
        y_label = y_var
        
        if x_transform:
            x_data = self.calculate_z_score(x_data)
            x_label = f"{x_var} (Z-score)"
        
        if y_transform:
            y_data = self.calculate_z_score(y_data)
            y_label = f"{y_var} (Z-score)"
        
        # Prepare size data
        size_data = None
        size_label = None
        if size_var:
            size_data = plot_df[size_var].astype(float).values
            # Normalize size data to a reasonable range (e.g., 5 to 25)
            size_min, size_max = 5, 25
            size_data = np.clip(size_data, np.percentile(size_data, 5), np.percentile(size_data, 95))
            size_data = size_min + (size_data - size_data.min()) * (size_max - size_min) / (size_data.max() - size_data.min() + 1e-10)
            size_label = size_var
        
        # Calculate statistics
        stats_info = self.calculate_statistics(x_data, y_data)
        self.analysis_results = stats_info
        
        # Create figure
        fig = go.Figure()
        
        # Identify outliers
        outlier_mask = np.zeros(len(x_data), dtype=bool)
        if show_outliers:
            if x_transform:
                outlier_mask |= np.abs(x_data) > 2
            if y_transform:
                outlier_mask |= np.abs(y_data) > 2
            if not x_transform and not y_transform:
                # Use Z-scores of original data for outlier detection
                x_z = self.calculate_z_score(x_original)
                y_z = self.calculate_z_score(y_original)
                outlier_mask = (np.abs(x_z) > 2) | (np.abs(y_z) > 2)
        
        # Add scatter points
        if color_var and color_var in plot_df.columns:
            # Color by category
            categories = plot_df[color_var].unique()
            colors = px.colors.qualitative.Set3[:len(categories)]
            
            for i, category in enumerate(categories):
                mask = plot_df[color_var] == category
                cat_outliers = outlier_mask & mask.values
                cat_normal = mask.values & ~outlier_mask
                
                # Normal points
                if np.any(cat_normal):
                    fig.add_trace(go.Scatter(
                        x=x_data[cat_normal],
                        y=y_data[cat_normal],
                        mode='markers',
                        name=str(category),
                        marker=dict(
                            size=size_data[cat_normal] if size_var else 8,
                            color=colors[i % len(colors)],
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{x_label}</b>: %{{x:.3f}}<br>' +
                                    f'<b>{y_label}</b>: %{{y:.3f}}<br>' +
                                    f'<b>{color_var}</b>: {category}' +
                                    (f'<br><b>{size_label}</b>: %{{marker.size:.3f}}' if size_var else '') +
                                    '<extra></extra>'
                    ))
                
                # Outlier points
                if show_outliers and np.any(cat_outliers):
                    fig.add_trace(go.Scatter(
                        x=x_data[cat_outliers],
                        y=y_data[cat_outliers],
                        mode='markers',
                        name=f'{category} (Outliers)',
                        marker=dict(
                            size=(size_data[cat_outliers] * 1.5) if size_var else 12,
                            color=colors[i % len(colors)],
                            opacity=0.9,
                            symbol='diamond',
                            line=dict(width=2, color='red')
                        ),
                        hovertemplate=f'<b>{x_label}</b>: %{{x:.3f}}<br>' +
                                    f'<b>{y_label}</b>: %{{y:.3f}}<br>' +
                                    f'<b>{color_var}</b>: {category}' +
                                    (f'<br><b>{size_label}</b>: %{{marker.size:.3f}}' if size_var else '') +
                                    '<br><b>Status</b>: Outlier<extra></extra>'
                    ))
        else:
            # Single color
            normal_mask = ~outlier_mask
            
            # Normal points
            if np.any(normal_mask):
                fig.add_trace(go.Scatter(
                    x=x_data[normal_mask],
                    y=y_data[normal_mask],
                    mode='markers',
                    name='Data Points',
                    marker=dict(
                        size=size_data[normal_mask] if size_var else 8,
                        color='steelblue',
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f'<b>{x_label}</b>: %{{x:.3f}}<br>' +
                                f'<b>{y_label}</b>: %{{y:.3f}}' +
                                (f'<br><b>{size_label}</b>: %{{marker.size:.3f}}' if size_var else '') +
                                '<extra></extra>'
                ))
            
            # Outlier points
            if show_outliers and np.any(outlier_mask):
                fig.add_trace(go.Scatter(
                    x=x_data[outlier_mask],
                    y=y_data[outlier_mask],
                    mode='markers',
                    name='Outliers',
                    marker=dict(
                        size=(size_data[outlier_mask] * 1.5) if size_var else 12,
                        color='red',
                        opacity=0.9,
                        symbol='diamond',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate=f'<b>{x_label}</b>: %{{x:.3f}}<br>' +
                                f'<b>{y_label}</b>: %{{y:.3f}}' +
                                (f'<br><b>{size_label}</b>: %{{marker.size:.3f}}' if size_var else '') +
                                '<br><b>Status</b>: Outlier<extra></extra>'
                ))
        
        # Add trend line
        if show_trend and stats_info:
            x_range = np.linspace(x_data.min(), x_data.max(), 100)
            y_trend = stats_info['slope'] * x_range + stats_info['intercept']
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_trend,
                mode='lines',
                name=f'Trend Line (R² = {stats_info["r_squared"]:.3f})',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Trend Line<extra></extra>'
            ))
        
        # Update layout
        title = f'Scatter Plot Analysis: {y_label} vs {x_label}'
        if self.file_path:
            title += f'<br><sub>Data: {os.path.basename(self.file_path)}</sub>'
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=700,
            showlegend=True
        )
        
        # Display plot in VS Code
        fig.show()
        
        # Print summary
        print(f"✅ Plot created successfully!")
        print(f"📊 Data points: {len(plot_df)}")
        print(f"⚠️ Outliers: {np.sum(outlier_mask)}")
        if stats_info:
            print(f"📈 Correlation: {stats_info['correlation']:.4f}")
            print(f"🎯 R-squared: {stats_info['r_squared']:.4f}")
        
        return fig
    
    def print_detailed_statistics(self):
        """Print comprehensive statistical analysis"""
        if not self.analysis_results:
            print("❌ No analysis results available. Create a plot first.")
            return
        
        stats = self.analysis_results
        
        print(f"\n📊 DETAILED STATISTICAL ANALYSIS")
        print("=" * 60)
        
        # Sample information
        print(f"📋 Sample Size: {stats['n_points']} data points")
        print()
        
        # Correlation Analysis
        print("🔗 CORRELATION ANALYSIS:")
        corr = stats['correlation']
        corr_abs = abs(corr)
        
        if corr_abs >= 0.8:
            strength = "Very Strong 💪"
        elif corr_abs >= 0.6:
            strength = "Strong 🔥"
        elif corr_abs >= 0.4:
            strength = "Moderate 📊"
        elif corr_abs >= 0.2:
            strength = "Weak 📉"
        else:
            strength = "Very Weak 😴"
        
        direction = "Positive ↗️" if corr > 0 else "Negative ↘️"
        
        print(f"  • Correlation Coefficient (r): {corr:.4f}")
        print(f"  • Strength: {strength}")
        print(f"  • Direction: {direction}")
        print(f"  • R-squared (r²): {stats['r_squared']:.4f}")
        print(f"  • Explained Variance: {stats['r_squared']*100:.1f}%")
        print()
        
        # Regression Analysis
        print("📈 LINEAR REGRESSION:")
        print(f"  • Equation: y = {stats['slope']:.4f}x + {stats['intercept']:.4f}")
        print(f"  • Slope: {stats['slope']:.4f}")
        print(f"  • Intercept: {stats['intercept']:.4f}")
        print(f"  • Standard Error: {stats['std_error']:.4f}")
        print(f"  • P-value: {stats['p_value']:.2e}")
        print()
        
        # Statistical Significance
        print("🎯 STATISTICAL SIGNIFICANCE:")
        alpha_levels = [0.001, 0.01, 0.05, 0.1]
        p_val = stats['p_value']
        
        for alpha in alpha_levels:
            if p_val < alpha:
                print(f"  ✅ Highly significant at α = {alpha} level")
                break
        else:
            print(f"  ❌ Not statistically significant at α = 0.1 level")
        
        print(f"  • P-value: {p_val:.2e}")
        print()
        
        # Descriptive Statistics
        print("📊 DESCRIPTIVE STATISTICS:")
        print(f"  X-axis: Mean = {stats['x_mean']:.4f}, SD = {stats['x_std']:.4f}")
        print(f"  Y-axis: Mean = {stats['y_mean']:.4f}, SD = {stats['y_std']:.4f}")
        print()
        
        # Interpretation Guide
        print("💡 INTERPRETATION GUIDE:")
        print("  • |r| > 0.8: Variables strongly related")
        print("  • p < 0.05: Relationship is statistically significant")
        print("  • R² shows % of variance explained by the relationship")
        print("  • Z-score > |2|: Data point is an outlier")
    
    def export_results(self, x_var, y_var, color_var=None, size_var=None):
        """Export analysis results to CSV"""
        if self.df is None:
            print("❌ No data to export")
            return
        
        try:
            # Prepare export data
            columns = [x_var, y_var]
            if color_var:
                columns.append(color_var)
            if size_var:
                columns.append(size_var)
            export_df = self.df[columns].copy()
            
            # Add Z-score columns
            export_df[f'{x_var}_zscore'] = self.calculate_z_score(export_df[x_var].dropna())
            export_df[f'{y_var}_zscore'] = self.calculate_z_score(export_df[y_var].dropna())
            
            # Add outlier flags
            x_outliers = np.abs(export_df[f'{x_var}_zscore']) > 2
            y_outliers = np.abs(export_df[f'{y_var}_zscore']) > 2
            export_df['is_outlier'] = x_outliers | y_outliers
            
            # Generate filename
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_file = f"{base_name}_analysis_{x_var}_vs_{y_var}.csv"
            
            # Save file
            export_df.to_csv(output_file, index=False)
            print(f"💾 Results exported to: {output_file}")
            
            # Export statistics summary
            if self.analysis_results:
                stats_file = f"{base_name}_statistics_{x_var}_vs_{y_var}.txt"
                with open(stats_file, 'w') as f:
                    f.write(f"Statistical Analysis Results\n")
                    f.write(f"==========================\n\n")
                    f.write(f"Variables: {y_var} vs {x_var}\n")
                    if color_var:
                        f.write(f"Color Variable: {color_var}\n")
                    if size_var:
                        f.write(f"Size Variable: {size_var}\n")
                    f.write(f"Data Source: {self.file_path}\n\n")
                    
                    stats = self.analysis_results
                    f.write(f"Sample Size: {stats['n_points']}\n")
                    f.write(f"Correlation: {stats['correlation']:.4f}\n")
                    f.write(f"R-squared: {stats['r_squared']:.4f}\n")
                    f.write(f"P-value: {stats['p_value']:.2e}\n")
                    f.write(f"Regression: y = {stats['slope']:.4f}x + {stats['intercept']:.4f}\n")
                
                print(f"📊 Statistics exported to: {stats_file}")
            
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
    
    def run_interactive_analysis(self):
        """Main interactive analysis workflow"""
        print("🚀 VS CODE EXCEL SCATTER PLOT ANALYZER")
        print("=" * 60)
        print("Features: Z-score transformation, outlier detection, size mapping, statistics")
        print()
        
        # Step 1: Load Excel file
        if not self.load_excel():
            return
        
        # Step 2: Select variables
        x_var, y_var, color_var, size_var = self.select_variables()
        if x_var is None or y_var is None:
            return
        
        # Step 3: Select transformations
        x_transform, y_transform, show_outliers, show_trend = self.select_transformations()
        
        # Step 4: Create plot
        fig = self.create_scatter_plot(
            x_var, y_var, color_var, size_var,
            x_transform, y_transform,
            show_outliers, show_trend
        )
        
        if fig is None:
            return
        
        # Step 5: Show detailed statistics
        self.print_detailed_statistics()
        
        # Step 6: Export option
        export_choice = input(f"\n💾 Export results to CSV? (y/N): ").lower().startswith('y')
        if export_choice:
            self.export_results(x_var, y_var, color_var, size_var)
        
        print(f"\n✅ Analysis complete! 🎉")

# Convenience functions for VS Code interactive use
def quick_scatter(file_path, x_col, y_col, color_col=None, size_col=None, z_x=False, z_y=False):
    """Quick scatter plot function for VS Code"""
    analyzer = ExcelScatterAnalyzer()
    if analyzer.load_excel(file_path):
        fig = analyzer.create_scatter_plot(x_col, y_col, color_col, size_col, z_x, z_y)
        analyzer.print_detailed_statistics()
        return analyzer
    return None

def create_sample_excel():
    """Create sample Excel file for testing"""
    np.random.seed(42)
    n = 150
    
    # Generate correlated sample data
    data = {
        'Height_cm': np.random.normal(170, 10, n),
        'Weight_kg': np.random.normal(70, 15, n),
        'Age_years': np.random.randint(18, 80, n),
        'Income_USD': np.random.lognormal(10, 0.8, n),
        'Education_years': np.random.randint(8, 20, n),
        'Category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Score': np.random.normal(75, 12, n)
    }
    
    # Add realistic correlations
    for i in range(n):
        # Weight correlates with height
        data['Weight_kg'][i] = data['Height_cm'][i] * 0.7 + np.random.normal(0, 8)
        # Income correlates with education
        data['Income_USD'][i] = np.exp(data['Education_years'][i] * 0.3 + np.random.normal(9, 0.5))
        # Score correlates with education and age
        data['Score'][i] = data['Education_years'][i] * 2 + data['Age_years'][i] * 0.2 + np.random.normal(30, 10)
    
    df = pd.DataFrame(data)
    filename = 'sample_data_for_analysis.xlsx'
    df.to_excel(filename, index=False)
    
    print(f"✅ Sample Excel file created: {filename}")
    print(f"📊 Data shape: {df.shape}")
    print("🔍 Suggested analyses:")
    print("   • Height_cm vs Weight_kg (with Category coloring, Age_years sizing)")
    print("   • Education_years vs Income_USD (log-transformed, Score sizing)")
    print("   • Age_years vs Score (with Z-score transformation, Income_USD sizing)")
    
    return filename

# Main execution
if __name__ == "__main__":
    print("🎯 VS Code Excel Scatter Plot Analyzer")
    print("\nChoose an option:")
    print("1. Interactive Analysis (recommended)")
    print("2. Create Sample Data")
    print("3. Quick Analysis (enter file path)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        analyzer = ExcelScatterAnalyzer()
        analyzer.run_interactive_analysis()
    
    elif choice == "2":
        filename = create_sample_excel()
        run_analysis = input(f"\nRun analysis on {filename}? (y/N): ").lower().startswith('y')
        if run_analysis:
            analyzer = ExcelScatterAnalyzer()
            analyzer.run_interactive_analysis()
    
    elif choice == "3":
        file_path = input("Enter Excel file path: ").strip()
        if os.path.exists(file_path):
            analyzer = ExcelScatterAnalyzer()
            analyzer.load_excel(file_path)
            analyzer.run_interactive_analysis()
        else:
            print("❌ File not found")
    
    else:
        print("❌ Invalid choice")