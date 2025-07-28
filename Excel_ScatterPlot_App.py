"""
Excel Scatter Plot App with Z-Score Transformation and Variable Point Sizing
============================================================================

A comprehensive interactive scatter plot analyzer for Excel data with advanced features:
- Z-score transformations for standardization
- Variable point sizing (aes-like mapping)
- Outlier detection and highlighting
- Statistical analysis and correlation metrics
- Interactive file selection and variable mapping
- Export capabilities with detailed statistics

Requirements: pip install pandas plotly numpy scipy openpyxl streamlit (optional for web UI)

Author: Data Analysis Tool
Version: 2.0
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

class ExcelScatterPlotApp:
    """
    Excel Scatter Plot Analyzer with Z-Score Transformation and Variable Point Sizing
    
    This class provides comprehensive scatter plot analysis capabilities including:
    - Interactive Excel file loading
    - Variable selection with aesthetic mappings (x, y, color, size)
    - Z-score transformations for data standardization
    - Outlier detection and visualization
    - Statistical analysis and regression
    - Export functionality
    """
    
    def __init__(self):
        """Initialize the scatter plot analyzer"""
        self.df = None
        self.file_path = None
        self.analysis_results = {}
        self.plot_config = {}
        
    def discover_excel_files(self, search_path=None, recursive=True):
        """
        Discover Excel files in specified directory
        
        Args:
            search_path (str): Directory to search (default: current directory)
            recursive (bool): Search subdirectories recursively
            
        Returns:
            list: List of Excel file paths found
        """
        if search_path is None:
            search_path = os.getcwd()
        
        excel_patterns = ['*.xlsx', '*.xls', '*.xlsm']
        excel_files = []
        
        print(f"🔍 Searching for Excel files in: {search_path}")
        
        for pattern in excel_patterns:
            if recursive:
                excel_files.extend(glob.glob(os.path.join(search_path, '**', pattern), recursive=True))
            else:
                excel_files.extend(glob.glob(os.path.join(search_path, pattern)))
        
        # Remove duplicates and sort
        excel_files = sorted(list(set(excel_files)))
        
        print(f"📁 Found {len(excel_files)} Excel files")
        return excel_files
    
    def interactive_file_selector(self):
        """
        Interactive Excel file selection with multiple options
        
        Returns:
            str: Selected file path or None if cancelled
        """
        print("\n" + "="*60)
        print("🎯 EXCEL FILE SELECTOR")
        print("="*60)
        
        # Option 1: Direct path entry
        print("\n📂 Option 1: Enter file path directly")
        direct_path = input("Enter Excel file path (or press Enter to browse): ").strip()
        
        if direct_path:
            if os.path.exists(direct_path) and direct_path.lower().endswith(('.xlsx', '.xls', '.xlsm')):
                return direct_path
            else:
                print("❌ Invalid file path or not an Excel file")
        
        # Option 2: Browse current directory
        print("\n📁 Option 2: Select from available files")
        excel_files = self.discover_excel_files()
        
        if not excel_files:
            print("❌ No Excel files found in current directory")
            
            # Option 3: Search different directory
            new_path = input("\n📍 Enter different directory to search (or press Enter to exit): ").strip()
            if new_path and os.path.exists(new_path):
                excel_files = self.discover_excel_files(new_path)
            
            if not excel_files:
                print("❌ No Excel files found")
                return None
        
        # Display found files
        print(f"\n📋 Available Excel Files:")
        print("-" * 70)
        
        for i, file_path in enumerate(excel_files, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / 1024  # KB
            mod_time = os.path.getmtime(file_path)
            from datetime import datetime
            mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            
            print(f"{i:2d}. {file_name:<30} ({file_size:6.1f} KB) [{mod_date}]")
        
        print("-" * 70)
        
        # File selection
        try:
            choice = input(f"\n🎯 Select file number (1-{len(excel_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                return None
                
            choice_num = int(choice)
            if 1 <= choice_num <= len(excel_files):
                selected_file = excel_files[choice_num - 1]
                print(f"\n✅ Selected: {os.path.basename(selected_file)}")
                return selected_file
            else:
                print("❌ Invalid selection")
                return None
                
        except ValueError:
            print("❌ Invalid input")
            return None
    
    def load_excel_data(self, file_path=None, sheet_name=None):
        """
        Load Excel data with comprehensive error handling and data preview
        
        Args:
            file_path (str): Path to Excel file
            sheet_name (str): Sheet name to load (default: first sheet)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if file_path is None:
            file_path = self.interactive_file_selector()
        
        if file_path is None:
            print("❌ No file selected")
            return False
        
        try:
            print(f"\n📊 Loading Excel file: {os.path.basename(file_path)}")
            
            # Check if file has multiple sheets
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1 and sheet_name is None:
                print(f"\n📋 Multiple sheets found:")
                for i, sheet in enumerate(sheet_names, 1):
                    print(f"{i}. {sheet}")
                
                try:
                    sheet_choice = input(f"\nSelect sheet (1-{len(sheet_names)}) or Enter for first sheet: ").strip()
                    if sheet_choice:
                        sheet_idx = int(sheet_choice) - 1
                        if 0 <= sheet_idx < len(sheet_names):
                            sheet_name = sheet_names[sheet_idx]
                        else:
                            print("❌ Invalid sheet selection, using first sheet")
                            sheet_name = sheet_names[0]
                    else:
                        sheet_name = sheet_names[0]
                except ValueError:
                    print("❌ Invalid input, using first sheet")
                    sheet_name = sheet_names[0]
            elif sheet_name is None:
                sheet_name = sheet_names[0]
            
            # Load the data
            self.df = pd.read_excel(file_path, sheet_name=sheet_name)
            self.file_path = file_path
            
            # Data summary
            print(f"\n✅ Successfully loaded sheet: '{sheet_name}'")
            print(f"📊 Dataset dimensions: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            
            # Column analysis
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
            
            print(f"🔢 Numeric columns: {len(numeric_cols)}")
            print(f"📝 Categorical columns: {len(categorical_cols)}")
            print(f"📅 DateTime columns: {len(datetime_cols)}")
            
            # Data quality check
            print(f"\n🔍 Data Quality Overview:")
            print("-" * 50)
            
            for i, col in enumerate(self.df.columns[:10], 1):  # Show first 10 columns
                dtype = str(self.df[col].dtype)
                null_count = self.df[col].isnull().sum()
                null_pct = (null_count / len(self.df)) * 100
                unique_count = self.df[col].nunique()
                
                print(f"{i:2d}. {col:<20} | {dtype:<12} | {null_count:4d} nulls ({null_pct:4.1f}%) | {unique_count:4d} unique")
            
            if len(self.df.columns) > 10:
                print(f"    ... and {len(self.df.columns) - 10} more columns")
            
            # Data preview
            print(f"\n📋 Data Preview (first 3 rows):")
            print("-" * 80)
            pd.set_option('display.max_columns', 6)
            pd.set_option('display.width', 80)
            print(self.df.head(3))
            pd.reset_option('display.max_columns')
            pd.reset_option('display.width')
            
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return False
        except PermissionError:
            print(f"❌ Permission denied: {file_path}")
            return False
        except Exception as e:
            print(f"❌ Error loading Excel file: {str(e)}")
            return False
    
    def select_plot_variables(self):
        """
        Interactive selection of variables for scatter plot with aesthetic mappings
        
        Returns:
            tuple: (x_var, y_var, color_var, size_var) or (None, None, None, None) if cancelled
        """
        if self.df is None:
            print("❌ No data loaded. Please load Excel file first.")
            return None, None, None, None
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols = self.df.columns.tolist()
        
        if len(numeric_cols) < 2:
            print("❌ Need at least 2 numeric columns for scatter plot")
            print(f"Available numeric columns: {numeric_cols}")
            return None, None, None, None
        
        print("\n" + "="*60)
        print("🎯 VARIABLE SELECTION & AESTHETIC MAPPING")
        print("="*60)
        print("Select variables for your scatter plot (similar to ggplot2 aes() mapping)")
        
        # X-axis variable selection
        print(f"\n📊 Select X-axis variable (required):")
        print("-" * 40)
        for i, col in enumerate(numeric_cols, 1):
            sample_vals = self.df[col].dropna().iloc[:3].values
            print(f"{i:2d}. {col:<25} (e.g., {sample_vals})")
        
        try:
            x_choice = int(input(f"\n🎯 Enter choice for X-axis (1-{len(numeric_cols)}): "))
            if 1 <= x_choice <= len(numeric_cols):
                x_var = numeric_cols[x_choice - 1]
                print(f"✅ X-axis: {x_var}")
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        # Y-axis variable selection
        print(f"\n📈 Select Y-axis variable (required):")
        print("-" * 40)
        for i, col in enumerate(numeric_cols, 1):
            marker = " ← Selected for X-axis" if col == x_var else ""
            sample_vals = self.df[col].dropna().iloc[:3].values
            print(f"{i:2d}. {col:<25} (e.g., {sample_vals}){marker}")
        
        try:
            y_choice = int(input(f"\n🎯 Enter choice for Y-axis (1-{len(numeric_cols)}): "))
            if 1 <= y_choice <= len(numeric_cols):
                y_var = numeric_cols[y_choice - 1]
                print(f"✅ Y-axis: {y_var}")
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        # Color variable selection (optional)
        print(f"\n🎨 Select color variable (optional - for grouping/categories):")
        print("-" * 50)
        print("0. None (single color)")
        for i, col in enumerate(all_cols, 1):
            markers = []
            if col == x_var:
                markers.append("X-axis")
            if col == y_var:
                markers.append("Y-axis")
            marker_text = f" ← {', '.join(markers)}" if markers else ""
            
            unique_count = self.df[col].nunique()
            col_type = "numeric" if col in numeric_cols else "categorical"
            print(f"{i:2d}. {col:<25} ({col_type}, {unique_count} unique){marker_text}")
        
        try:
            color_choice = int(input(f"\n🎯 Enter choice for color (0-{len(all_cols)}): "))
            if color_choice == 0:
                color_var = None
                print("✅ Color: Single color (no grouping)")
            elif 1 <= color_choice <= len(all_cols):
                color_var = all_cols[color_choice - 1]
                print(f"✅ Color: {color_var}")
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        # Size variable selection (optional)
        print(f"\n📏 Select size variable (optional - for variable point sizing):")
        print("-" * 55)
        print("0. None (fixed size)")
        for i, col in enumerate(numeric_cols, 1):
            markers = []
            if col == x_var:
                markers.append("X-axis")
            if col == y_var:
                markers.append("Y-axis")
            if col == color_var:
                markers.append("Color")
            marker_text = f" ← {', '.join(markers)}" if markers else ""
            
            sample_vals = self.df[col].dropna().iloc[:3].values
            print(f"{i:2d}. {col:<25} (e.g., {sample_vals}){marker_text}")
        
        try:
            size_choice = int(input(f"\n🎯 Enter choice for size (0-{len(numeric_cols)}): "))
            if size_choice == 0:
                size_var = None
                print("✅ Size: Fixed size")
            elif 1 <= size_choice <= len(numeric_cols):
                size_var = numeric_cols[size_choice - 1]
                print(f"✅ Size: {size_var}")
            else:
                print("❌ Invalid choice")
                return None, None, None, None
        except ValueError:
            print("❌ Invalid input")
            return None, None, None, None
        
        # Summary of selections
        print(f"\n📋 Variable Mapping Summary:")
        print("-" * 40)
        print(f"X-axis:  {x_var}")
        print(f"Y-axis:  {y_var}")
        print(f"Color:   {color_var if color_var else 'None'}")
        print(f"Size:    {size_var if size_var else 'Fixed'}")
        
        return x_var, y_var, color_var, size_var
    
    def configure_transformations(self, x_var, y_var):
        """
        Configure Z-score transformations and plot options
        
        Args:
            x_var (str): X-axis variable name
            y_var (str): Y-axis variable name
            
        Returns:
            dict: Configuration dictionary
        """
        print("\n" + "="*60)
        print("🔄 TRANSFORMATION & PLOT CONFIGURATION")
        print("="*60)
        
        print("\n📊 Z-Score Transformation Options:")
        print("Z-score formula: Z = (X - μ) / σ")
        print("Benefits:")
        print("  • Standardizes variables to same scale (mean=0, std=1)")
        print("  • Makes different units comparable")
        print("  • Helps identify outliers (|Z| > 2 or 3)")
        print("  • Useful for correlation analysis")
        
        # X-axis transformation
        print(f"\n🔄 Transform X-axis ({x_var}) to Z-scores?")
        x_stats = self.df[x_var].describe()
        print(f"   Current range: {x_stats['min']:.2f} to {x_stats['max']:.2f}")
        print(f"   Mean: {x_stats['mean']:.2f}, Std: {x_stats['std']:.2f}")
        x_transform = input("   Apply Z-score transformation? (y/N): ").lower().startswith('y')
        
        # Y-axis transformation
        print(f"\n🔄 Transform Y-axis ({y_var}) to Z-scores?")
        y_stats = self.df[y_var].describe()
        print(f"   Current range: {y_stats['min']:.2f} to {y_stats['max']:.2f}")
        print(f"   Mean: {y_stats['mean']:.2f}, Std: {y_stats['std']:.2f}")
        y_transform = input("   Apply Z-score transformation? (y/N): ").lower().startswith('y')
        
        # Outlier detection
        print(f"\n⚠️  Outlier Detection & Highlighting:")
        print("   Outliers defined as |Z-score| > 2 (95% of normal data)")
        show_outliers = input("   Highlight outliers with special markers? (Y/n): ").lower() != 'n'
        
        # Trend line
        print(f"\n📈 Trend Line Options:")
        print("   Adds linear regression line with R² value")
        show_trend = input("   Show trend line? (Y/n): ").lower() != 'n'
        
        # Additional plot options
        print(f"\n🎨 Additional Plot Options:")
        show_marginals = input("   Show marginal distributions? (y/N): ").lower().startswith('y')
        show_correlation = input("   Display correlation coefficient on plot? (Y/n): ").lower() != 'n'
        
        config = {
            'x_transform': x_transform,
            'y_transform': y_transform,
            'show_outliers': show_outliers,
            'show_trend': show_trend,
            'show_marginals': show_marginals,
            'show_correlation': show_correlation
        }
        
        return config
    
    def calculate_z_score(self, data, method='standard'):
        """
        Calculate Z-score with multiple methods
        
        Args:
            data (array-like): Input data
            method (str): 'standard', 'robust', or 'modified'
            
        Returns:
            numpy.array: Z-scores
        """
        data = np.array(data)
        
        if method == 'standard':
            # Standard Z-score: (X - μ) / σ
            return (data - np.nanmean(data)) / np.nanstd(data, ddof=1)
        elif method == 'robust':
            # Robust Z-score using median and MAD
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            return 0.6745 * (data - median) / mad
        elif method == 'modified':
            # Modified Z-score (less sensitive to outliers)
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            return 0.6745 * (data - median) / mad
        else:
            raise ValueError("Method must be 'standard', 'robust', or 'modified'")
    
    def normalize_sizes(self, size_data, min_size=6, max_size=30):
        """
        Normalize size values for point sizing with better scaling
        
        Args:
            size_data (array-like): Size variable data
            min_size (int): Minimum point size
            max_size (int): Maximum point size
            
        Returns:
            numpy.array: Normalized sizes
        """
        size_data = np.array(size_data, dtype=float)
        
        # Handle missing values
        valid_mask = ~np.isnan(size_data)
        if not np.any(valid_mask):
            return np.full_like(size_data, (min_size + max_size) / 2)
        
        valid_data = size_data[valid_mask]
        
        # Use percentile-based normalization to handle outliers
        p5, p95 = np.percentile(valid_data, [5, 95])
        
        if p95 == p5:
            # All values are essentially the same
            return np.full_like(size_data, (min_size + max_size) / 2)
        
        # Clip extreme values and normalize
        clipped_data = np.clip(size_data, p5, p95)
        normalized = min_size + (clipped_data - p5) / (p95 - p5) * (max_size - min_size)
        
        # Handle NaN values
        normalized = np.where(np.isnan(normalized), (min_size + max_size) / 2, normalized)
        
        return normalized
    
    def calculate_comprehensive_statistics(self, x_data, y_data):
        """
        Calculate comprehensive statistical measures
        
        Args:
            x_data (array-like): X variable data
            y_data (array-like): Y variable data
            
        Returns:
            dict: Statistical results
        """
        # Clean data
        x_clean = np.array(x_data, dtype=float)
        y_clean = np.array(y_data, dtype=float)
        
        # Remove infinite and NaN values
        valid_mask = np.isfinite(x_clean) & np.isfinite(y_clean)
        x_clean = x_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        
        if len(x_clean) < 3:
            return None
        
        # Basic statistics
        try:
            # Correlation analysis
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            
            # Descriptive statistics
            x_stats = {
                'mean': np.mean(x_clean),
                'std': np.std(x_clean, ddof=1),
                'median': np.median(x_clean),
                'min': np.min(x_clean),
                'max': np.max(x_clean),
                'q25': np.percentile(x_clean, 25),
                'q75': np.percentile(x_clean, 75)
            }
            
            y_stats = {
                'mean': np.mean(y_clean),
                'std': np.std(y_clean, ddof=1),
                'median': np.median(y_clean),
                'min': np.min(y_clean),
                'max': np.max(y_clean),
                'q25': np.percentile(y_clean, 25),
                'q75': np.percentile(y_clean, 75)
            }
            
            # Additional correlation measures
            spearman_corr, spearman_p = stats.spearmanr(x_clean, y_clean)
            kendall_corr, kendall_p = stats.kendalltau(x_clean, y_clean)
            
            return {
                'n_points': len(x_clean),
                'correlation': correlation,
                'r_squared': r_value**2,
                'slope': slope,
                'intercept': intercept,
                'p_value': p_value,
                'std_error': std_err,
                'x_stats': x_stats,
                'y_stats': y_stats,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'kendall_corr': kendall_corr,
                'kendall_p': kendall_p
            }
            
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate some statistics: {str(e)}")
            return None
    
    def create_enhanced_scatter_plot(self, x_var, y_var, color_var=None, size_var=None, config=None):
        """
        Create an enhanced scatter plot with all features
        
        Args:
            x_var (str): X-axis variable
            y_var (str): Y-axis variable  
            color_var (str): Color grouping variable
            size_var (str): Size mapping variable
            config (dict): Plot configuration
            
        Returns:
            plotly.graph_objects.Figure: The created plot
        """
        if self.df is None:
            print("❌ No data loaded")
            return None
        
        if config is None:
            config = {
                'x_transform': False,
                'y_transform': False,
                'show_outliers': True,
                'show_trend': True,
                'show_marginals': False,
                'show_correlation': True
            }
        
        print(f"\n🎨 Creating enhanced scatter plot...")
        print(f"   📊 Variables: {y_var} vs {x_var}")
        if color_var:
            print(f"   🎨 Color mapping: {color_var}")
        if size_var:
            print(f"   📏 Size mapping: {size_var}")
        
        # Prepare data
        required_cols = [x_var, y_var]
        if color_var:
            required_cols.append(color_var)
        if size_var:
            required_cols.append(size_var)
        
        plot_df = self.df[required_cols].dropna()
        
        if len(plot_df) == 0:
            print("❌ No valid data points after removing missing values")
            return None
        
        print(f"   📋 Valid data points: {len(plot_df):,}")
        
        # Get data arrays
        x_data = plot_df[x_var].astype(float).values
        y_data = plot_df[y_var].astype(float).values
        
        # Store original data for statistics and hover info
        x_original = x_data.copy()
        y_original = y_data.copy()
        
        # Apply transformations
        x_label = x_var
        y_label = y_var
        
        if config['x_transform']:
            x_data = self.calculate_z_score(x_data)
            x_label = f"{x_var} (Z-score)"
            print(f"   🔄 Applied Z-score transformation to X-axis")
        
        if config['y_transform']:
            y_data = self.calculate_z_score(y_data)
            y_label = f"{y_var} (Z-score)"
            print(f"   🔄 Applied Z-score transformation to Y-axis")
        
        # Calculate statistics
        stats_results = self.calculate_comprehensive_statistics(x_data, y_data)
        self.analysis_results = stats_results
        
        # Handle size mapping
        if size_var:
            size_data = plot_df[size_var].astype(float).values
            normalized_sizes = self.normalize_sizes(size_data)
            print(f"   📏 Size range: {plot_df[size_var].min():.2f} to {plot_df[size_var].max():.2f}")
        else:
            normalized_sizes = np.full(len(x_data), 10)  # Default size
        
        # Create figure (with or without marginals)
        if config['show_marginals']:
            fig = make_subplots(
                rows=2, cols=2,
                column_widths=[0.8, 0.2],
                row_heights=[0.2, 0.8],
                horizontal_spacing=0.02,
                vertical_spacing=0.02,
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            main_row, main_col = 2, 1
        else:
            fig = go.Figure()
            main_row, main_col = None, None
        
        # Identify outliers
        outlier_mask = np.zeros(len(x_data), dtype=bool)
        if config['show_outliers']:
            if config['x_transform'] or config['y_transform']:
                # Use transformed data for outlier detection
                x_z = x_data if config['x_transform'] else self.calculate_z_score(x_original)
                y_z = y_data if config['y_transform'] else self.calculate_z_score(y_original)
                outlier_mask = (np.abs(x_z) > 2) | (np.abs(y_z) > 2)
            else:
                # Use Z-scores of original data
                x_z = self.calculate_z_score(x_original)
                y_z = self.calculate_z_score(y_original)
                outlier_mask = (np.abs(x_z) > 2) | (np.abs(y_z) > 2)
        
        # Color mapping
        if color_var and color_var in plot_df.columns:
            self._add_colored_scatter_traces(
                fig, x_data, y_data, plot_df, color_var, size_var,
                normalized_sizes, outlier_mask, x_label, y_label,
                config, main_row, main_col
            )
        else:
            self._add_single_color_scatter_traces(
                fig, x_data, y_data, size_var, normalized_sizes,
                outlier_mask, x_label, y_label, config, main_row, main_col
            )
        
        # Add trend line
        if config['show_trend'] and stats_results:
            self._add_trend_line(fig, x_data, y_data, stats_results, main_row, main_col)
        
        # Add marginal distributions
        if config['show_marginals']:
            self._add_marginal_distributions(fig, x_data, y_data, x_label, y_label)
        
        # Configure layout
        self._configure_plot_layout(
            fig, x_label, y_label, color_var, size_var, 
            stats_results, config, main_row, main_col
        )
        
        # Store plot configuration
        self.plot_config = {
            'x_var': x_var,
            'y_var': y_var,
            'color_var': color_var,
            'size_var': size_var,
            'config': config
        }
        
        print(f"   ✅ Plot created successfully!")
        if config['show_outliers']:
            print(f"   ⚠️  Outliers detected: {np.sum(outlier_mask)}")
        if stats_results:
            print(f"   📈 Correlation: {stats_results['correlation']:.4f}")
            print(f"   🎯 R-squared: {stats_results['r_squared']:.4f}")
        
        # Display the plot
        fig.show()
        
        return fig
    
    def _add_colored_scatter_traces(self, fig, x_data, y_data, plot_df, color_var, 
                                   size_var, normalized_sizes, outlier_mask, 
                                   x_label, y_label, config, main_row, main_col):
        """Add scatter traces with color grouping"""
        categories = plot_df[color_var].unique()
        colors = px.colors.qualitative.Set3[:len(categories)]
        
        for i, category in enumerate(categories):
            mask = plot_df[color_var] == category
            cat_outliers = outlier_mask & mask.values
            cat_normal = mask.values & ~outlier_mask
            
            # Prepare hover data
            hover_data = self._prepare_hover_data(
                plot_df, mask, x_label, y_label, color_var, size_var, category
            )
            
            # Normal points
            if np.any(cat_normal):
                trace = go.Scatter(
                    x=x_data[cat_normal],
                    y=y_data[cat_normal],
                    mode='markers',
                    name=str(category),
                    marker=dict(
                        size=normalized_sizes[cat_normal],
                        color=colors[i % len(colors)],
                        opacity=0.7,
                        line=dict(width=1, color='white'),
                        sizemode='diameter'
                    ),
                    customdata=hover_data['normal'] if hover_data else None,
                    hovertemplate=hover_data['template'] if hover_data else None
                )
                
                if main_row and main_col:
                    fig.add_trace(trace, row=main_row, col=main_col)
                else:
                    fig.add_trace(trace)
            
            # Outlier points
            if config['show_outliers'] and np.any(cat_outliers):
                outlier_trace = go.Scatter(
                    x=x_data[cat_outliers],
                    y=y_data[cat_outliers],
                    mode='markers',
                    name=f'{category} (Outliers)',
                    marker=dict(
                        size=normalized_sizes[cat_outliers] * 1.3,
                        color=colors[i % len(colors)],
                        opacity=0.9,
                        symbol='diamond',
                        line=dict(width=2, color='red'),
                        sizemode='diameter'
                    ),
                    customdata=hover_data['outliers'] if hover_data else None,
                    hovertemplate=hover_data['outlier_template'] if hover_data else None
                )
                
                if main_row and main_col:
                    fig.add_trace(outlier_trace, row=main_row, col=main_col)
                else:
                    fig.add_trace(outlier_trace)
    
    def _add_single_color_scatter_traces(self, fig, x_data, y_data, size_var, 
                                        normalized_sizes, outlier_mask, 
                                        x_label, y_label, config, main_row, main_col):
        """Add scatter traces with single color"""
        normal_mask = ~outlier_mask
        
        # Prepare hover data
        hover_template = f'<b>{x_label}</b>: %{{x:.3f}}<br><b>{y_label}</b>: %{{y:.3f}}'
        if size_var:
            hover_template += f'<br><b>{size_var}</b>: %{{customdata:.3f}}'
        hover_template += '<extra></extra>'
        
        # Normal points
        if np.any(normal_mask):
            trace = go.Scatter(
                x=x_data[normal_mask],
                y=y_data[normal_mask],
                mode='markers',
                name='Data Points',
                marker=dict(
                    size=normalized_sizes[normal_mask],
                    color='steelblue',
                    opacity=0.7,
                    line=dict(width=1, color='white'),
                    sizemode='diameter'
                ),
                hovertemplate=hover_template
            )
            
            if main_row and main_col:
                fig.add_trace(trace, row=main_row, col=main_col)
            else:
                fig.add_trace(trace)
        
        # Outlier points
        if config['show_outliers'] and np.any(outlier_mask):
            outlier_hover = f'<b>{x_label}</b>: %{{x:.3f}}<br><b>{y_label}</b>: %{{y:.3f}}<br><b>Status</b>: Outlier'
            if size_var:
                outlier_hover += f'<br><b>{size_var}</b>: %{{customdata:.3f}}'
            outlier_hover += '<extra></extra>'
            
            outlier_trace = go.Scatter(
                x=x_data[outlier_mask],
                y=y_data[outlier_mask],
                mode='markers',
                name='Outliers',
                marker=dict(
                    size=normalized_sizes[outlier_mask] * 1.3,
                    color='red',
                    opacity=0.9,
                    symbol='diamond',
                    line=dict(width=2, color='darkred'),
                    sizemode='diameter'
                ),
                hovertemplate=outlier_hover
            )
            
            if main_row and main_col:
                fig.add_trace(outlier_trace, row=main_row, col=main_col)
            else:
                fig.add_trace(outlier_trace)
    
    def _prepare_hover_data(self, plot_df, mask, x_label, y_label, color_var, size_var, category):
        """Prepare hover data for colored traces"""
        base_template = f'<b>{x_label}</b>: %{{x:.3f}}<br><b>{y_label}</b>: %{{y:.3f}}<br>' + \
                       f'<b>{color_var}</b>: {category}'
        
        if size_var:
            base_template += f'<br><b>{size_var}</b>: %{{customdata:.3f}}'
            customdata = plot_df[size_var].values[mask]
        else:
            customdata = None
        
        return {
            'normal': customdata,
            'outliers': customdata,
            'template': base_template + '<extra></extra>',
            'outlier_template': base_template + '<br><b>Status</b>: Outlier<extra></extra>'
        }
    
    def _add_trend_line(self, fig, x_data, y_data, stats_results, main_row, main_col):
        """Add trend line to the plot"""
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_trend = stats_results['slope'] * x_range + stats_results['intercept']
        
        trend_trace = go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Trend Line (R² = {stats_results["r_squared"]:.3f})',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>Trend Line</b><br>R² = ' + 
                         f'{stats_results["r_squared"]:.3f}<extra></extra>'
        )
        
        if main_row and main_col:
            fig.add_trace(trend_trace, row=main_row, col=main_col)
        else:
            fig.add_trace(trend_trace)
    
    def _add_marginal_distributions(self, fig, x_data, y_data, x_label, y_label):
        """Add marginal distribution plots"""
        # X marginal (top)
        fig.add_trace(
            go.Histogram(x=x_data, name='X Distribution', showlegend=False, 
                        marker_color='lightblue', opacity=0.7),
            row=1, col=1
        )
        
        # Y marginal (right)
        fig.add_trace(
            go.Histogram(y=y_data, name='Y Distribution', showlegend=False,
                        marker_color='lightcoral', opacity=0.7),
            row=2, col=2
        )
    
    def _configure_plot_layout(self, fig, x_label, y_label, color_var, size_var, 
                              stats_results, config, main_row, main_col):
        """Configure the plot layout"""
        # Create title
        title_parts = [f'Scatter Plot Analysis: {y_label} vs {x_label}']
        
        if color_var:
            title_parts.append(f'Color: {color_var}')
        if size_var:
            title_parts.append(f'Size: {size_var}')
        
        title = '<br>'.join(title_parts)
        
        if self.file_path:
            title += f'<br><sub>Data: {os.path.basename(self.file_path)}</sub>'
        
        # Add correlation info to title if requested
        if config['show_correlation'] and stats_results:
            corr = stats_results['correlation']
            title += f'<br><sub>Correlation: {corr:.4f}</sub>'
        
        # Configure layout
        layout_config = dict(
            title=title,
            hovermode='closest',
            template='plotly_white',
            width=1000,
            height=700,
            showlegend=True
        )
        
        if not config['show_marginals']:
            layout_config.update({
                'xaxis_title': x_label,
                'yaxis_title': y_label
            })
        else:
            # Configure axes for subplot layout
            fig.update_xaxes(title_text=x_label, row=2, col=1)
            fig.update_yaxes(title_text=y_label, row=2, col=1)
            fig.update_xaxes(showticklabels=False, row=1, col=1)
            fig.update_yaxes(showticklabels=False, row=2, col=2)
        
        fig.update_layout(**layout_config)
    
    def display_comprehensive_statistics(self):
        """Display comprehensive statistical analysis"""
        if not self.analysis_results:
            print("❌ No analysis results available. Create a plot first.")
            return
        
        stats = self.analysis_results
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        # Sample information
        print(f"\n📋 Sample Information:")
        print(f"   • Sample Size: {stats['n_points']:,} data points")
        print(f"   • Data Source: {os.path.basename(self.file_path) if self.file_path else 'Unknown'}")
        
        # Correlation Analysis
        print(f"\n🔗 Correlation Analysis:")
        corr = stats['correlation']
        corr_abs = abs(corr)
        
        # Correlation strength interpretation
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
        
        print(f"   • Pearson Correlation (r): {corr:.4f}")
        print(f"   • Strength: {strength}")
        print(f"   • Direction: {direction}")
        print(f"   • R-squared (r²): {stats['r_squared']:.4f}")
        print(f"   • Explained Variance: {stats['r_squared']*100:.1f}%")
        
        # Additional correlation measures
        print(f"   • Spearman Correlation: {stats['spearman_corr']:.4f} (p = {stats['spearman_p']:.3e})")
        print(f"   • Kendall's Tau: {stats['kendall_corr']:.4f} (p = {stats['kendall_p']:.3e})")
        
        # Regression Analysis
        print(f"\n📈 Linear Regression Analysis:")
        print(f"   • Equation: y = {stats['slope']:.4f}x + {stats['intercept']:.4f}")
        print(f"   • Slope: {stats['slope']:.4f}")
        print(f"   • Intercept: {stats['intercept']:.4f}")
        print(f"   • Standard Error: {stats['std_error']:.4f}")
        print(f"   • P-value: {stats['p_value']:.2e}")
        
        # Statistical Significance
        print(f"\n🎯 Statistical Significance:")
        p_val = stats['p_value']
        alpha_levels = [0.001, 0.01, 0.05, 0.1]
        
        significance_found = False
        for alpha in alpha_levels:
            if p_val < alpha:
                print(f"   ✅ Significant at α = {alpha} level")
                significance_found = True
                break
        
        if not significance_found:
            print(f"   ❌ Not statistically significant at α = 0.1 level")
        
        print(f"   • P-value: {p_val:.2e}")
        
        # Descriptive Statistics
        print(f"\n📊 Descriptive Statistics:")
        x_stats = stats['x_stats']
        y_stats = stats['y_stats']
        
        print(f"   X-axis Variable:")
        print(f"     - Mean: {x_stats['mean']:.4f}, Std: {x_stats['std']:.4f}")
        print(f"     - Median: {x_stats['median']:.4f}")
        print(f"     - Range: {x_stats['min']:.4f} to {x_stats['max']:.4f}")
        print(f"     - IQR: {x_stats['q25']:.4f} to {x_stats['q75']:.4f}")
        
        print(f"   Y-axis Variable:")
        print(f"     - Mean: {y_stats['mean']:.4f}, Std: {y_stats['std']:.4f}")
        print(f"     - Median: {y_stats['median']:.4f}")
        print(f"     - Range: {y_stats['min']:.4f} to {y_stats['max']:.4f}")
        print(f"     - IQR: {y_stats['q25']:.4f} to {y_stats['q75']:.4f}")
        
        # Interpretation Guide
        print(f"\n💡 Interpretation Guide:")
        print(f"   • |r| > 0.8: Variables are strongly related")
        print(f"   • p < 0.05: Relationship is statistically significant")
        print(f"   • R² shows percentage of variance explained by the relationship")
        print(f"   • Spearman correlation measures monotonic relationships")
        print(f"   • Z-score > |2|: Data point is considered an outlier")
        print(f"   • Point sizes reflect the magnitude of the size variable")
    
    def export_analysis_results(self, filename=None):
        """Export analysis results and data to files"""
        if self.df is None:
            print("❌ No data to export")
            return False
        
        if not self.plot_config:
            print("❌ No plot configuration found. Create a plot first.")
            return False
        
        try:
            config = self.plot_config
            x_var = config['x_var']
            y_var = config['y_var']
            color_var = config['color_var']
            size_var = config['size_var']
            
            # Prepare export data
            required_cols = [x_var, y_var]
            if color_var:
                required_cols.append(color_var)
            if size_var:
                required_cols.append(size_var)
            
            export_df = self.df[required_cols].copy()
            
            # Add transformed variables
            if config['config']['x_transform']:
                export_df[f'{x_var}_zscore'] = self.calculate_z_score(export_df[x_var])
            if config['config']['y_transform']:
                export_df[f'{y_var}_zscore'] = self.calculate_z_score(export_df[y_var])
            
            # Add outlier detection
            x_z = self.calculate_z_score(export_df[x_var])
            y_z = self.calculate_z_score(export_df[y_var])
            export_df['is_outlier'] = (np.abs(x_z) > 2) | (np.abs(y_z) > 2)
            export_df['x_zscore_abs'] = np.abs(x_z)
            export_df['y_zscore_abs'] = np.abs(y_z)
            
            # Add normalized sizes if size variable is used
            if size_var:
                size_data = export_df[size_var].astype(float).values
                export_df[f'{size_var}_normalized_size'] = self.normalize_sizes(size_data)
            
            # Generate filenames
            base_name = os.path.splitext(os.path.basename(self.file_path))[0] if self.file_path else "analysis"
            
            if filename is None:
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                data_file = f"{base_name}_scatter_analysis_{x_var}_vs_{y_var}_{timestamp}.csv"
                stats_file = f"{base_name}_statistics_{x_var}_vs_{y_var}_{timestamp}.txt"
            else:
                data_file = f"{filename}_data.csv"
                stats_file = f"{filename}_statistics.txt"
            
            # Export data
            export_df.to_csv(data_file, index=False)
            print(f"💾 Data exported to: {data_file}")
            
            # Export statistics
            if self.analysis_results:
                self._export_statistics_report(stats_file)
                print(f"📊 Statistics exported to: {stats_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
            return False
    
    def _export_statistics_report(self, filename):
        """Export detailed statistics report to text file"""
        stats = self.analysis_results
        config = self.plot_config
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Excel Scatter Plot Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.file_path}\n")
            f.write(f"Variables: {config['y_var']} vs {config['x_var']}\n")
            
            if config['color_var']:
                f.write(f"Color Variable: {config['color_var']}\n")
            if config['size_var']:
                f.write(f"Size Variable: {config['size_var']}\n")
            
            f.write(f"\nTransformations Applied:\n")
            f.write(f"  X-axis Z-score: {config['config']['x_transform']}\n")
            f.write(f"  Y-axis Z-score: {config['config']['y_transform']}\n")
            f.write(f"  Outlier detection: {config['config']['show_outliers']}\n")
            
            f.write(f"\nSample Statistics:\n")
            f.write(f"  Sample Size: {stats['n_points']}\n")
            
            f.write(f"\nCorrelation Analysis:\n")
            f.write(f"  Pearson Correlation: {stats['correlation']:.6f}\n")
            f.write(f"  Spearman Correlation: {stats['spearman_corr']:.6f}\n")
            f.write(f"  Kendall's Tau: {stats['kendall_corr']:.6f}\n")
            f.write(f"  R-squared: {stats['r_squared']:.6f}\n")
            
            f.write(f"\nRegression Analysis:\n")
            f.write(f"  Slope: {stats['slope']:.6f}\n")
            f.write(f"  Intercept: {stats['intercept']:.6f}\n")
            f.write(f"  Standard Error: {stats['std_error']:.6f}\n")
            f.write(f"  P-value: {stats['p_value']:.2e}\n")
            
            f.write(f"\nDescriptive Statistics:\n")
            x_stats = stats['x_stats']
            y_stats = stats['y_stats']
            
            f.write(f"  X-axis ({config['x_var']}):\n")
            for key, value in x_stats.items():
                f.write(f"    {key}: {value:.6f}\n")
            
            f.write(f"  Y-axis ({config['y_var']}):\n")
            for key, value in y_stats.items():
                f.write(f"    {key}: {value:.6f}\n")
    
    def create_comparison_plots(self):
        """Create comparison plots showing different aspects of the data"""
        if not self.plot_config:
            print("❌ No plot configuration found. Create a plot first.")
            return None
        
        config = self.plot_config
        x_var = config['x_var']
        y_var = config['y_var']
        size_var = config['size_var']
        
        if not size_var:
            print("❌ Size variable required for comparison plots")
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Without Size Mapping',
                'With Size Mapping', 
                'Size Distribution',
                'Correlation Matrix'
            ),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # Get clean data
        plot_df = self.df[[x_var, y_var, size_var]].dropna()
        x_data = plot_df[x_var].values
        y_data = plot_df[y_var].values
        size_data = plot_df[size_var].values
        normalized_sizes = self.normalize_sizes(size_data)
        
        # Plot 1: Without size mapping
        fig.add_trace(
            go.Scatter(
                x=x_data, y=y_data,
                mode='markers',
                name='Fixed Size',
                marker=dict(size=8, color='steelblue', opacity=0.7),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Plot 2: With size mapping
        fig.add_trace(
            go.Scatter(
                x=x_data, y=y_data,
                mode='markers',
                name='Variable Size',
                marker=dict(
                    size=normalized_sizes,
                    color=size_data,
                    colorscale='Viridis',
                    opacity=0.7,
                    sizemode='diameter',
                    showscale=True,
                    colorbar=dict(title=size_var)
                ),
                showlegend=True
            ),
            row=1, col=2
        )
        
        # Plot 3: Size distribution
        fig.add_trace(
            go.Histogram(
                x=size_data,
                name='Size Distribution',
                marker_color='lightgreen',
                opacity=0.7,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Plot 4: Correlation heatmap
        corr_data = plot_df[[x_var, y_var, size_var]].corr()
        fig.add_trace(
            go.Heatmap(
                z=corr_data.values,
                x=corr_data.columns,
                y=corr_data.columns,
                colorscale='RdBu',
                zmin=-1, zmax=1,
                text=corr_data.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'Comparison Analysis: {y_var} vs {x_var} (Size: {size_var})',
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text=x_var, row=1, col=1)
        fig.update_xaxes(title_text=x_var, row=1, col=2)
        fig.update_xaxes(title_text=size_var, row=2, col=1)
        fig.update_yaxes(title_text=y_var, row=1, col=1)
        fig.update_yaxes(title_text=y_var, row=1, col=2)
        fig.update_yaxes(title_text='Count', row=2, col=1)
        
        fig.show()
        return fig
    
    def run_complete_analysis(self):
        """Run the complete interactive analysis workflow"""
        print("🚀 EXCEL SCATTER PLOT ANALYZER")
        print("=" * 80)
        print("Features:")
        print("  • Z-score transformations for data standardization")
        print("  • Variable point sizing (aes-like mapping)")
        print("  • Outlier detection and highlighting")
        print("  • Comprehensive statistical analysis")
        print("  • Interactive visualizations with Plotly")
        print("  • Export capabilities")
        print("=" * 80)
        
        # Step 1: Load Excel data
        print("\n🔶 STEP 1: DATA LOADING")
        if not self.load_excel_data():
            print("❌ Analysis terminated: Could not load data")
            return False
        
        # Step 2: Variable selection
        print("\n🔶 STEP 2: VARIABLE SELECTION")
        x_var, y_var, color_var, size_var = self.select_plot_variables()
        if x_var is None or y_var is None:
            print("❌ Analysis terminated: Invalid variable selection")
            return False
        
        # Step 3: Configuration
        print("\n🔶 STEP 3: PLOT CONFIGURATION")
        config = self.configure_transformations(x_var, y_var)
        
        # Step 4: Create plot
        print("\n🔶 STEP 4: PLOT GENERATION")
        fig = self.create_enhanced_scatter_plot(x_var, y_var, color_var, size_var, config)
        
        if fig is None:
            print("❌ Analysis terminated: Could not create plot")
            return False
        
        # Step 5: Statistical analysis
        print("\n🔶 STEP 5: STATISTICAL ANALYSIS")
        self.display_comprehensive_statistics()
        
        # Step 6: Additional options
        print("\n🔶 STEP 6: ADDITIONAL OPTIONS")
        
        # Comparison plots
        if size_var:
            show_comparison = input("📊 Create comparison plots? (y/N): ").lower().startswith('y')
            if show_comparison:
                self.create_comparison_plots()
        
        # Export results
        export_choice = input("💾 Export results to files? (y/N): ").lower().startswith('y')
        if export_choice:
            custom_name = input("Enter custom filename (or press Enter for auto): ").strip()
            filename = custom_name if custom_name else None
            self.export_analysis_results(filename)
        
        print("\n✅ ANALYSIS COMPLETE! 🎉")
        print("=" * 80)
        
        return True


# Utility functions for quick analysis
def quick_analysis(file_path, x_col, y_col, color_col=None, size_col=None, 
                  z_x=False, z_y=False, show_stats=True):
    """
    Quick scatter plot analysis function
    
    Args:
        file_path (str): Path to Excel file
        x_col (str): X-axis variable name
        y_col (str): Y-axis variable name
        color_col (str): Color grouping variable name
        size_col (str): Size mapping variable name
        z_x (bool): Apply Z-score to X-axis
        z_y (bool): Apply Z-score to Y-axis
        show_stats (bool): Display statistics
    
    Returns:
        ExcelScatterPlotApp: Analyzer instance
    """
    analyzer = ExcelScatterPlotApp()
    
    if analyzer.load_excel_data(file_path):
        config = {
            'x_transform': z_x,
            'y_transform': z_y,
            'show_outliers': True,
            'show_trend': True,
            'show_marginals': False,
            'show_correlation': True
        }
        
        fig = analyzer.create_enhanced_scatter_plot(x_col, y_col, color_col, size_col, config)
        
        if show_stats and fig:
            analyzer.display_comprehensive_statistics()
        
        return analyzer
    
    return None


def create_sample_dataset():
    """
    Create a comprehensive sample Excel dataset for testing
    
    Returns:
        str: Filename of created Excel file
    """
    np.random.seed(42)
    n = 200
    
    print("🔧 Creating comprehensive sample dataset...")
    
    # Generate realistic correlated data
    data = {
        # Physical measurements
        'Height_cm': np.random.normal(170, 10, n),
        'Weight_kg': np.random.normal(70, 15, n),
        'BMI': [],  # Will be calculated
        
        # Demographics
        'Age_years': np.random.randint(18, 80, n),
        'Education_years': np.random.randint(8, 20, n),
        'Experience_years': [],  # Will be calculated
        
        # Economic data
        'Income_USD': np.random.lognormal(10, 0.8, n),
        'Savings_USD': [],  # Will be calculated
        
        # Performance metrics
        'Test_Score': np.random.normal(75, 12, n),
        'Performance_Rating': np.random.normal(7, 1.5, n),
        
        # Categorical variables
        'Department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n),
        'Education_Level': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n),
        'City_Size': np.random.choice(['Small', 'Medium', 'Large', 'Mega'], n),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n),
        
        # Geographic/Scale variables
        'Population': np.random.lognormal(12, 1.2, n),  # City population
        'Budget_USD': np.random.lognormal(8, 0.6, n),  # Project budgets
        'Team_Size': np.random.randint(3, 25, n),
        
        # Time-based
        'Years_in_Company': np.random.randint(1, 30, n),
        'Projects_Completed': np.random.randint(5, 100, n),
        
        # Quality metrics
        'Customer_Satisfaction': np.random.normal(8, 1.2, n),
        'Error_Rate': np.random.exponential(0.05, n),
        'Efficiency_Score': np.random.normal(85, 10, n)
    }
    
    # Add realistic correlations and calculated fields
    for i in range(n):
        # Weight correlates with height (BMI calculation)
        data['Weight_kg'][i] = data['Height_cm'][i] * 0.7 + np.random.normal(0, 8)
        data['BMI'].append(data['Weight_kg'][i] / ((data['Height_cm'][i] / 100) ** 2))
        
        # Experience correlates with age but capped
        max_exp = max(0, data['Age_years'][i] - 22)  # Assuming work starts at 22
        actual_exp = min(max_exp, np.random.randint(0, max_exp + 5) if max_exp > 0 else 0)
        data['Experience_years'].append(actual_exp)
        
        # Income correlates with education, experience, and performance
        income_factor = (
            data['Education_years'][i] * 0.15 +
            actual_exp * 0.08 +
            data['Performance_Rating'][i] * 0.2 +
            np.random.normal(8.5, 0.7)
        )
        data['Income_USD'][i] = np.exp(income_factor)
        
        # Savings correlate with income and age
        savings_rate = 0.1 + (data['Age_years'][i] - 25) * 0.002  # Increases with age
        savings_rate = max(0.05, min(0.3, savings_rate))  # Cap between 5% and 30%
        data['Savings_USD'].append(data['Income_USD'][i] * savings_rate * np.random.uniform(0.5, 1.5))
        
        # Test score correlates with education and age (experience)
        education_bonus = data['Education_years'][i] * 1.5
        experience_bonus = actual_exp * 0.3
        base_score = 50 + education_bonus + experience_bonus + np.random.normal(0, 8)
        data['Test_Score'][i] = max(0, min(100, base_score))
        
        # Performance rating correlates with test score and experience
        perf_base = (
            data['Test_Score'][i] * 0.05 +
            actual_exp * 0.08 +
            np.random.normal(5, 1)
        )
        data['Performance_Rating'][i] = max(1, min(10, perf_base))
        
        # Customer satisfaction correlates with performance and experience
        cust_sat = (
            data['Performance_Rating'][i] * 0.6 +
            actual_exp * 0.05 +
            np.random.normal(3, 0.8)
        )
        data['Customer_Satisfaction'][i] = max(1, min(10, cust_sat))
        
        # Error rate inversely correlates with experience and performance
        error_base = 0.15 - (actual_exp * 0.003) - (data['Performance_Rating'][i] * 0.008)
        data['Error_Rate'][i] = max(0.001, error_base + np.random.exponential(0.02))
        
        # Efficiency correlates with performance and inversely with error rate
        eff_score = (
            data['Performance_Rating'][i] * 5 +
            50 - (data['Error_Rate'][i] * 200) +
            np.random.normal(0, 5)
        )
        data['Efficiency_Score'][i] = max(20, min(100, eff_score))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some interesting categorical mappings
    dept_mapping = {
        'Engineering': {'avg_income_mult': 1.2, 'avg_education': 16},
        'Marketing': {'avg_income_mult': 1.0, 'avg_education': 14},
        'Sales': {'avg_income_mult': 1.1, 'avg_education': 13},
        'HR': {'avg_income_mult': 0.9, 'avg_education': 15},
        'Finance': {'avg_income_mult': 1.15, 'avg_education': 16}
    }
    
    # Apply department-based adjustments
    for dept, props in dept_mapping.items():
        mask = df['Department'] == dept
        df.loc[mask, 'Income_USD'] *= props['avg_income_mult']
        # Slight education bias by department
        education_adj = np.random.normal(props['avg_education'] - 14, 1, mask.sum())
        df.loc[mask, 'Education_years'] = np.clip(
            df.loc[mask, 'Education_years'] + education_adj, 8, 20
        ).astype(int)
    
    # Create filename with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f'comprehensive_sample_data_{timestamp}.xlsx'
    
    # Save to Excel
    df.to_excel(filename, index=False)
    
    print(f"✅ Sample dataset created: {filename}")
    print(f"📊 Dataset dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"🔢 Numeric variables: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"📝 Categorical variables: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Display analysis suggestions
    print(f"\n🎯 Suggested Analysis Combinations:")
    print(f"=" * 60)
    
    suggestions = [
        {
            'title': 'Physical Health Analysis',
            'x': 'Height_cm', 'y': 'Weight_kg', 'color': 'Gender', 'size': 'Age_years',
            'description': 'Explore height-weight relationship by gender, sized by age'
        },
        {
            'title': 'Career Development Analysis', 
            'x': 'Experience_years', 'y': 'Income_USD', 'color': 'Department', 'size': 'Education_years',
            'description': 'Income growth with experience across departments'
        },
        {
            'title': 'Performance vs Education',
            'x': 'Education_years', 'y': 'Performance_Rating', 'color': 'Education_Level', 'size': 'Team_Size',
            'description': 'Education impact on performance, sized by team responsibility'
        },
        {
            'title': 'Efficiency Analysis',
            'x': 'Error_Rate', 'y': 'Efficiency_Score', 'color': 'Department', 'size': 'Experience_years',
            'description': 'Error rates vs efficiency across departments'
        },
        {
            'title': 'Customer Success Factors',
            'x': 'Performance_Rating', 'y': 'Customer_Satisfaction', 'color': 'City_Size', 'size': 'Projects_Completed',
            'description': 'Performance impact on customer satisfaction'
        },
        {
            'title': 'Economic Analysis',
            'x': 'Income_USD', 'y': 'Savings_USD', 'color': 'Education_Level', 'size': 'Age_years',
            'description': 'Savings patterns by income and education level'
        }
    ]
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion['title']}:")
        print(f"   X: {suggestion['x']}, Y: {suggestion['y']}")
        print(f"   Color: {suggestion['color']}, Size: {suggestion['size']}")
        print(f"   → {suggestion['description']}")
        print()
    
    # Display sample usage code
    print(f"💻 Sample Usage Code:")
    print(f"=" * 60)
    print(f"# Quick analysis example:")
    print(f"analyzer = quick_analysis(")
    print(f"    '{filename}',")
    print(f"    x_col='Experience_years',")
    print(f"    y_col='Income_USD',")
    print(f"    color_col='Department',")
    print(f"    size_col='Education_years',")
    print(f"    z_x=False, z_y=True  # Z-score transform income")
    print(f")")
    print()
    print(f"# Or run interactive analysis:")
    print(f"app = ExcelScatterPlotApp()")
    print(f"app.run_complete_analysis()")
    
    return filename


def batch_analysis(file_path, variable_combinations, output_dir=None):
    """
    Perform batch analysis on multiple variable combinations
    
    Args:
        file_path (str): Path to Excel file
        variable_combinations (list): List of dicts with 'x', 'y', 'color', 'size' keys
        output_dir (str): Output directory for exports
    
    Returns:
        dict: Results dictionary with analysis summaries
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    analyzer = ExcelScatterPlotApp()
    if not analyzer.load_excel_data(file_path):
        return None
    
    results = {}
    
    print(f"🔄 Running batch analysis on {len(variable_combinations)} combinations...")
    
    for i, combo in enumerate(variable_combinations, 1):
        print(f"\n📊 Analysis {i}/{len(variable_combinations)}: {combo.get('title', f'Combo {i}')}")
        
        try:
            config = {
                'x_transform': combo.get('z_x', False),
                'y_transform': combo.get('z_y', False),
                'show_outliers': True,
                'show_trend': True,
                'show_marginals': False,
                'show_correlation': True
            }
            
            fig = analyzer.create_enhanced_scatter_plot(
                combo['x'], combo['y'], 
                combo.get('color'), combo.get('size'),
                config
            )
            
            if fig and analyzer.analysis_results:
                # Store results
                combo_key = f"{combo['y']}_vs_{combo['x']}"
                results[combo_key] = {
                    'combination': combo,
                    'statistics': analyzer.analysis_results.copy(),
                    'correlation': analyzer.analysis_results['correlation'],
                    'r_squared': analyzer.analysis_results['r_squared'],
                    'p_value': analyzer.analysis_results['p_value']
                }
                
                # Export if output directory specified
                if output_dir:
                    export_name = os.path.join(output_dir, f"analysis_{i:02d}_{combo_key}")
                    analyzer.export_analysis_results(export_name)
                
                print(f"   ✅ Correlation: {analyzer.analysis_results['correlation']:.4f}")
                print(f"   📈 R²: {analyzer.analysis_results['r_squared']:.4f}")
                
        except Exception as e:
            print(f"   ❌ Error in analysis {i}: {str(e)}")
            continue
    
    # Create summary report
    if results:
        print(f"\n📋 BATCH ANALYSIS SUMMARY")
        print(f"=" * 60)
        
        # Sort by correlation strength
        sorted_results = sorted(results.items(), 
                              key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        print(f"{'Rank':<4} {'Variables':<35} {'Correlation':<12} {'R²':<8} {'P-value'}")
        print(f"-" * 70)
        
        for rank, (key, result) in enumerate(sorted_results, 1):
            combo = result['combination']
            corr = result['correlation']
            r2 = result['r_squared']
            p_val = result['p_value']
            
            var_desc = f"{combo['y']} vs {combo['x']}"
            if len(var_desc) > 35:
                var_desc = var_desc[:32] + "..."
            
            print(f"{rank:<4} {var_desc:<35} {corr:>8.4f} {r2:>8.4f} {p_val:>8.2e}")
    
    return results


# Main execution
if __name__ == "__main__":
    print("🎯 EXCEL SCATTER PLOT ANALYZER WITH Z-SCORE & VARIABLE SIZING")
    print("=" * 80)
    print("Choose an option:")
    print("1. 🚀 Interactive Complete Analysis (Recommended)")
    print("2. 📊 Create Comprehensive Sample Dataset") 
    print("3. ⚡ Quick Analysis (Enter parameters)")
    print("4. 🔄 Batch Analysis (Multiple combinations)")
    print("5. 📈 Demo with Sample Data")
    print("6. ❓ Help & Documentation")
    
    try:
        choice = input(f"\n🎯 Enter your choice (1-6): ").strip()
        
        if choice == "1":
            # Interactive complete analysis
            app = ExcelScatterPlotApp()
            app.run_complete_analysis()
            
        elif choice == "2":
            # Create sample dataset
            filename = create_sample_dataset()
            
            run_analysis = input(f"\n🚀 Run interactive analysis on {filename}? (Y/n): ")
            if run_analysis.lower() != 'n':
                app = ExcelScatterPlotApp()
                app.load_excel_data(filename)
                app.run_complete_analysis()
                
        elif choice == "3":
            # Quick analysis
            file_path = input("📁 Enter Excel file path: ").strip()
            if not os.path.exists(file_path):
                print("❌ File not found")
            else:
                print("\n📊 Available columns (loading file to check):")
                temp_app = ExcelScatterPlotApp()
                if temp_app.load_excel_data(file_path):
                    numeric_cols = temp_app.df.select_dtypes(include=[np.number]).columns.tolist()
                    all_cols = temp_app.df.columns.tolist()
                    
                    print("Numeric columns:", ", ".join(numeric_cols[:10]))
                    print("All columns:", ", ".join(all_cols[:10]))
                    
                    x_col = input("🔵 X-axis variable: ").strip()
                    y_col = input("🔴 Y-axis variable: ").strip()
                    color_col = input("🎨 Color variable (optional): ").strip() or None
                    size_col = input("📏 Size variable (optional): ").strip() or None
                    
                    z_x = input("🔄 Z-score transform X? (y/N): ").lower().startswith('y')
                    z_y = input("🔄 Z-score transform Y? (y/N): ").lower().startswith('y')
                    
                    analyzer = quick_analysis(file_path, x_col, y_col, color_col, size_col, z_x, z_y)
                    
        elif choice == "4":
            # Batch analysis
            file_path = input("📁 Enter Excel file path: ").strip()
            if not os.path.exists(file_path):
                print("❌ File not found")
            else:
                print("📋 Enter variable combinations (format: x_var,y_var,color_var,size_var)")
                print("    Example: Height_cm,Weight_kg,Gender,Age_years")
                print("    Press Enter on empty line to finish")
                
                combinations = []
                i = 1
                while True:
                    combo_input = input(f"Combination {i}: ").strip()
                    if not combo_input:
                        break
                    
                    parts = [p.strip() for p in combo_input.split(',')]
                    if len(parts) >= 2:
                        combo = {
                            'x': parts[0],
                            'y': parts[1],
                            'color': parts[2] if len(parts) > 2 and parts[2] else None,
                            'size': parts[3] if len(parts) > 3 and parts[3] else None,
                            'title': f"{parts[1]} vs {parts[0]}"
                        }
                        combinations.append(combo)
                        i += 1
                    else:
                        print("❌ Invalid format. Need at least x,y variables")
                
                if combinations:
                    output_dir = input("📁 Output directory (optional): ").strip() or None
                    batch_analysis(file_path, combinations, output_dir)
                    
        elif choice == "5":
            # Demo with sample data
            print("🎬 Creating demo dataset and running sample analyses...")
            filename = create_sample_dataset()
            
            # Predefined demo combinations
            demo_combinations = [
                {
                    'title': 'Career Analysis',
                    'x': 'Experience_years', 'y': 'Income_USD', 
                    'color': 'Department', 'size': 'Education_years'
                },
                {
                    'title': 'Performance Analysis',
                    'x': 'Education_years', 'y': 'Performance_Rating',
                    'color': 'Education_Level', 'size': 'Team_Size'
                }
            ]
            
            print(f"\n🚀 Running demo analyses...")
            results = batch_analysis(filename, demo_combinations)
            
        elif choice == "6":
            # Help & Documentation
            print("\n📚 EXCEL SCATTER PLOT ANALYZER - HELP & DOCUMENTATION")
            print("=" * 70)
            print("\n🎯 Purpose:")
            print("   Advanced scatter plot analysis tool for Excel data with:")
            print("   • Z-score transformations for standardization")
            print("   • Variable point sizing (ggplot2-style aes mapping)")
            print("   • Outlier detection and statistical analysis")
            print("   • Interactive visualizations with Plotly")
            
            print("\n📊 Key Features:")
            print("   • Interactive file and variable selection")
            print("   • Multiple correlation measures (Pearson, Spearman, Kendall)")
            print("   • Comprehensive statistical analysis")
            print("   • Export capabilities (CSV + statistics report)")
            print("   • Batch processing for multiple variable combinations")
            print("   • Sample data generation for testing")
            
            print("\n🔧 Quick Start:")
            print("   1. Prepare Excel file with numeric variables")
            print("   2. Run option 1 for interactive analysis")
            print("   3. Select variables for X, Y, color, and size mapping")
            print("   4. Configure Z-score transformations")
            print("   5. Analyze results and export if needed")
            
            print("\n💡 Best Practices:")
            print("   • Use Z-score transformation when variables have different scales")
            print("   • Choose size variable that adds meaningful insight")
            print("   • Color variable should have reasonable number of categories (<10)")
            print("   • Check for outliers and consider their impact")
            print("   • Export results for reproducibility")
            
        else:
            print("❌ Invalid choice. Please select 1-6.")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Please try again or contact support.")