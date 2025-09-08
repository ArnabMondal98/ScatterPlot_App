import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import numpy as np
import warnings
from pathlib import Path
import os
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure page - moved inside main function to avoid context issues
def configure_page():
    try:
        st.set_page_config(
            page_title="Advanced BI Dashboard - Data Visualization",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except st.errors.StreamlitAPIException:
        # Page config already set, ignore
        pass

def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
            border-bottom: 3px solid #1f77b4;
            margin-bottom: 2rem;
        }
        
        .metric-container {
            background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin: 1rem 0;
        }
        
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .upload-section {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px dashed #dee2e6;
            margin: 1rem 0;
        }
        
        .analysis-card {
            background: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
            border-left: 4px solid #28a745;
        }
        
        .correlation-strong {
            color: #28a745;
            font-weight: bold;
        }
        
        .correlation-moderate {
            color: #ffc107;
            font-weight: bold;
        }
        
        .correlation-weak {
            color: #dc3545;
            font-weight: bold;
        }
        
        /* Hide Streamlit menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

class AdvancedScatterPlotAnalyzer:
    """Advanced Scatter Plot Analyzer with comprehensive statistical analysis"""
    
    def __init__(self):
        self.df = None
        self.analysis_results = {}
        self.plot_config = {}
    
    def calculate_z_score(self, data, method='standard'):
        """Calculate Z-score with multiple methods"""
        data = np.array(data, dtype=float)
        
        # Remove infinite and NaN values for calculation
        valid_mask = np.isfinite(data)
        if not np.any(valid_mask):
            return np.full_like(data, np.nan)
        
        valid_data = data[valid_mask]
        
        if method == 'standard':
            # Standard Z-score: (X - μ) / σ
            if len(valid_data) < 2:
                return np.full_like(data, np.nan)
            z_scores = np.full_like(data, np.nan, dtype=float)
            z_scores[valid_mask] = (valid_data - np.mean(valid_data)) / np.std(valid_data, ddof=1)
            return z_scores
        elif method == 'robust':
            # Robust Z-score using median and MAD
            median = np.median(valid_data)
            mad = np.median(np.abs(valid_data - median))
            if mad == 0:
                return np.full_like(data, np.nan)
            z_scores = np.full_like(data, np.nan, dtype=float)
            z_scores[valid_mask] = 0.6745 * (valid_data - median) / mad
            return z_scores
        else:
            raise ValueError("Method must be 'standard' or 'robust'")
    
    def normalize_sizes(self, size_data, min_size=6, max_size=30):
        """Normalize size values for point sizing with better scaling"""
        size_data = np.array(size_data, dtype=float)
        
        # Handle missing values
        valid_mask = ~np.isnan(size_data)
        if not np.any(valid_mask):
            return np.full_like(size_data, (min_size + max_size) / 2)
        
        valid_data = size_data[valid_mask]
        
        # Use percentile-based normalization to handle outliers
        if len(valid_data) < 2:
            return np.full_like(size_data, (min_size + max_size) / 2)
        
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
        """Calculate comprehensive statistical measures"""
        # Clean data
        x_clean = np.array(x_data, dtype=float)
        y_clean = np.array(y_data, dtype=float)
        
        # Remove infinite and NaN values
        valid_mask = np.isfinite(x_clean) & np.isfinite(y_clean)
        x_clean = x_clean[valid_mask]
        y_clean = y_clean[valid_mask]
        
        if len(x_clean) < 3:
            return None
        
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
            st.warning(f"Could not calculate some statistics: {str(e)}")
            return None

def load_data(uploaded_file):
    """Backend function to load and process data from uploaded spreadsheet"""
    try:
        if uploaded_file is not None:
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Read CSV
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                # Read Excel
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return None
            
            # Basic data cleaning
            df = df.dropna(how='all')  # Remove completely empty rows
            df.columns = df.columns.str.strip()  # Remove whitespace from column names
            
            return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_numeric_columns(df):
    """Get numeric columns from dataframe for visualization"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_columns

def create_enhanced_scatter_plot(df, x_var, y_var, color_var=None, size_var=None, 
                               x_transform=False, y_transform=False, 
                               show_outliers=True, show_trend=True, 
                               show_marginals=False, bubble_size_max=60):
    """Create an enhanced scatter plot with all advanced features"""
    
    analyzer = AdvancedScatterPlotAnalyzer()
    analyzer.df = df
    
    # Prepare data
    required_cols = [x_var, y_var]
    if color_var and color_var != "None":
        required_cols.append(color_var)
    if size_var and size_var != "None":
        required_cols.append(size_var)
    
    plot_df = df[required_cols].dropna()
    
    if len(plot_df) == 0:
        st.error("No valid data points after removing missing values")
        return None, None
    
    # Get data arrays
    x_data = plot_df[x_var].astype(float).values
    y_data = plot_df[y_var].astype(float).values
    
    # Store original data for statistics and hover info
    x_original = x_data.copy()
    y_original = y_data.copy()
    
    # Apply transformations
    x_label = x_var
    y_label = y_var
    
    if x_transform:
        x_data = analyzer.calculate_z_score(x_data)
        x_label = f"{x_var} (Z-score)"
    
    if y_transform:
        y_data = analyzer.calculate_z_score(y_data)
        y_label = f"{y_var} (Z-score)"
    
    # Calculate statistics
    stats_results = analyzer.calculate_comprehensive_statistics(x_data, y_data)
    
    # Handle size mapping
    if size_var and size_var != "None":
        size_data = plot_df[size_var].astype(float).values
        normalized_sizes = analyzer.normalize_sizes(size_data, min_size=6, max_size=bubble_size_max)
    else:
        normalized_sizes = np.full(len(x_data), 10)  # Default size
    
    # Create figure (with or without marginals)
    if show_marginals:
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
    if show_outliers:
        if x_transform or y_transform:
            # Use transformed data for outlier detection
            x_z = x_data if x_transform else analyzer.calculate_z_score(x_original)
            y_z = y_data if y_transform else analyzer.calculate_z_score(y_original)
            outlier_mask = (np.abs(x_z) > 2) | (np.abs(y_z) > 2)
        else:
            # Use Z-scores of original data
            x_z = analyzer.calculate_z_score(x_original)
            y_z = analyzer.calculate_z_score(y_original)
            outlier_mask = (np.abs(x_z) > 2) | (np.abs(y_z) > 2)
    
    # Color mapping
    if color_var and color_var != "None" and color_var in plot_df.columns:
        # Colored scatter traces
        categories = plot_df[color_var].unique()
        colors = px.colors.qualitative.Set3[:len(categories)]
        
        for i, category in enumerate(categories):
            mask = plot_df[color_var] == category
            cat_outliers = outlier_mask & mask.values
            cat_normal = mask.values & ~outlier_mask
            
            # Prepare hover template
            hover_template = f'<b>{x_label}</b>: %{{x:.3f}}<br><b>{y_label}</b>: %{{y:.3f}}<br>' + \
                           f'<b>{color_var}</b>: {category}'
            if size_var and size_var != "None":
                hover_template += f'<br><b>{size_var}</b>: %{{customdata:.3f}}'
            hover_template += '<extra></extra>'
            
            # Normal points
            if np.any(cat_normal):
                customdata = plot_df[size_var].values[mask] if size_var and size_var != "None" else None
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
                    customdata=customdata[cat_normal] if customdata is not None else None,
                    hovertemplate=hover_template
                )
                
                if main_row and main_col:
                    fig.add_trace(trace, row=main_row, col=main_col)
                else:
                    fig.add_trace(trace)
            
            # Outlier points
            if show_outliers and np.any(cat_outliers):
                outlier_hover = hover_template.replace('<extra></extra>', '<br><b>Status</b>: Outlier<extra></extra>')
                customdata = plot_df[size_var].values[mask] if size_var and size_var != "None" else None
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
                    customdata=customdata[cat_outliers] if customdata is not None else None,
                    hovertemplate=outlier_hover
                )
                
                if main_row and main_col:
                    fig.add_trace(outlier_trace, row=main_row, col=main_col)
                else:
                    fig.add_trace(outlier_trace)
    else:
        # Single color scatter traces
        normal_mask = ~outlier_mask
        
        # Prepare hover template
        hover_template = f'<b>{x_label}</b>: %{{x:.3f}}<br><b>{y_label}</b>: %{{y:.3f}}'
        if size_var and size_var != "None":
            hover_template += f'<br><b>{size_var}</b>: %{{customdata:.3f}}'
        hover_template += '<extra></extra>'
        
        # Normal points
        if np.any(normal_mask):
            customdata = plot_df[size_var].values if size_var and size_var != "None" else None
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
                customdata=customdata[normal_mask] if customdata is not None else None,
                hovertemplate=hover_template
            )
            
            if main_row and main_col:
                fig.add_trace(trace, row=main_row, col=main_col)
            else:
                fig.add_trace(trace)
        
        # Outlier points
        if show_outliers and np.any(outlier_mask):
            outlier_hover = hover_template.replace('<extra></extra>', '<br><b>Status</b>: Outlier<extra></extra>')
            customdata = plot_df[size_var].values if size_var and size_var != "None" else None
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
                customdata=customdata[outlier_mask] if customdata is not None else None,
                hovertemplate=outlier_hover
            )
            
            if main_row and main_col:
                fig.add_trace(outlier_trace, row=main_row, col=main_col)
            else:
                fig.add_trace(outlier_trace)
    
    # Add trend line
    if show_trend and stats_results:
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_trend = stats_results['slope'] * x_range + stats_results['intercept']
        
        trend_trace = go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Trend Line (R² = {stats_results["r_squared"]:.3f})',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate=f'<b>Trend Line</b><br>R² = {stats_results["r_squared"]:.3f}<extra></extra>'
        )
        
        if main_row and main_col:
            fig.add_trace(trend_trace, row=main_row, col=main_col)
        else:
            fig.add_trace(trend_trace)
    
    # Add marginal distributions
    if show_marginals:
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
    
    # Configure layout
    title_parts = [f'Advanced Scatter Plot: {y_label} vs {x_label}']
    
    if color_var and color_var != "None":
        title_parts.append(f'Color: {color_var}')
    if size_var and size_var != "None":
        title_parts.append(f'Size: {size_var}')
    
    title = '<br>'.join(title_parts)
    
    # Add correlation info to title
    if stats_results:
        corr = stats_results['correlation']
        title += f'<br><sub>Correlation: {corr:.4f} | R²: {stats_results["r_squared"]:.4f}</sub>'
    
    # Configure layout
    layout_config = dict(
        title=title,
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=700,
        showlegend=True
    )
    
    if not show_marginals:
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
    
    return fig, stats_results

def display_data_summary(df):
    """Display data summary statistics"""
    st.markdown("### 📈 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h4>Total Rows</h4>
            <h2 style="color: #1f77b4;">{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h4>Total Columns</h4>
            <h2 style="color: #ff7f0e;">{}</h2>
        </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    
    with col3:
        numeric_cols = len(get_numeric_columns(df))
        st.markdown("""
        <div class="metric-container">
            <h4>Numeric Columns</h4>
            <h2 style="color: #2ca02c;">{}</h2>
        </div>
        """.format(numeric_cols), unsafe_allow_html=True)
    
    with col4:
        missing_values = df.isnull().sum().sum()
        st.markdown("""
        <div class="metric-container">
            <h4>Missing Values</h4>
            <h2 style="color: #d62728;">{}</h2>
        </div>
        """.format(missing_values), unsafe_allow_html=True)

def display_comprehensive_statistics(stats_results, x_var, y_var):
    """Display comprehensive statistical analysis"""
    if not stats_results:
        st.warning("No statistical analysis available.")
        return
    
    st.markdown("### 📊 Comprehensive Statistical Analysis")
    
    # Create analysis cards
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation Analysis
        corr = stats_results['correlation']
        corr_abs = abs(corr)
        
        # Correlation strength interpretation
        if corr_abs >= 0.8:
            strength = "Very Strong 💪"
            strength_class = "correlation-strong"
        elif corr_abs >= 0.6:
            strength = "Strong 🔥"
            strength_class = "correlation-strong"
        elif corr_abs >= 0.4:
            strength = "Moderate 📊"
            strength_class = "correlation-moderate"
        elif corr_abs >= 0.2:
            strength = "Weak 📉"
            strength_class = "correlation-weak"
        else:
            strength = "Very Weak 😴"
            strength_class = "correlation-weak"
        
        direction = "Positive ↗️" if corr > 0 else "Negative ↘️"
        
        st.markdown(f"""
        <div class="analysis-card">
            <h4>🔗 Correlation Analysis</h4>
            <p><strong>Pearson Correlation:</strong> {corr:.4f}</p>
            <p><strong>Strength:</strong> <span class="{strength_class}">{strength}</span></p>
            <p><strong>Direction:</strong> {direction}</p>
            <p><strong>R-squared:</strong> {stats_results['r_squared']:.4f}</p>
            <p><strong>Explained Variance:</strong> {stats_results['r_squared']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Regression Analysis
        p_val = stats_results['p_value']
        significance = "✅ Statistically Significant" if p_val < 0.05 else "❌ Not Significant"
        
        st.markdown(f"""
        <div class="analysis-card">
            <h4>📈 Linear Regression</h4>
            <p><strong>Equation:</strong> y = {stats_results['slope']:.4f}x + {stats_results['intercept']:.4f}</p>
            <p><strong>Slope:</strong> {stats_results['slope']:.4f}</p>
            <p><strong>P-value:</strong> {p_val:.2e}</p>
            <p><strong>Significance:</strong> {significance}</p>
            <p><strong>Sample Size:</strong> {stats_results['n_points']:,} points</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional correlation measures
    with st.expander("🔍 Advanced Statistical Measures", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Spearman Correlation", f"{stats_results['spearman_corr']:.4f}")
            st.caption(f"P-value: {stats_results['spearman_p']:.3e}")
        
        with col2:
            st.metric("Kendall's Tau", f"{stats_results['kendall_corr']:.4f}")
            st.caption(f"P-value: {stats_results['kendall_p']:.3e}")
        
        with col3:
            st.metric("Standard Error", f"{stats_results['std_error']:.4f}")
    
    # Descriptive Statistics
    with st.expander("📊 Descriptive Statistics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{x_var} Statistics:**")
            x_stats = stats_results['x_stats']
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Median', 'Min', 'Max', 'Q25', 'Q75'],
                'Value': [
                    f"{x_stats['mean']:.4f}",
                    f"{x_stats['std']:.4f}",
                    f"{x_stats['median']:.4f}",
                    f"{x_stats['min']:.4f}",
                    f"{x_stats['max']:.4f}",
                    f"{x_stats['q25']:.4f}",
                    f"{x_stats['q75']:.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.markdown(f"**{y_var} Statistics:**")
            y_stats = stats_results['y_stats']
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Std Dev', 'Median', 'Min', 'Max', 'Q25', 'Q75'],
                'Value': [
                    f"{y_stats['mean']:.4f}",
                    f"{y_stats['std']:.4f}",
                    f"{y_stats['median']:.4f}",
                    f"{y_stats['min']:.4f}",
                    f"{y_stats['max']:.4f}",
                    f"{y_stats['q25']:.4f}",
                    f"{y_stats['q75']:.4f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)

def create_sample_dataset():
    """Create a comprehensive sample dataset for testing"""
    np.random.seed(42)
    n = 200
    
    # Generate realistic correlated data
    data = {
        # Physical measurements
        'Height_cm': np.random.normal(170, 10, n),
        'Weight_kg': [],
        'BMI': [],
        
        # Demographics
        'Age_years': np.random.randint(18, 80, n),
        'Education_years': np.random.randint(8, 20, n),
        'Experience_years': [],
        
        # Economic data
        'Income_USD': [],
        'Savings_USD': [],
        
        # Performance metrics
        'Test_Score': np.random.normal(75, 12, n),
        'Performance_Rating': np.random.normal(7, 1.5, n),
        
        # Categorical variables
        'Department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'], n),
        'Education_Level': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n),
        'City_Size': np.random.choice(['Small', 'Medium', 'Large', 'Mega'], n),
        'Gender': np.random.choice(['Male', 'Female', 'Other'], n),
        
        # Geographic/Scale variables
        'Population': np.random.lognormal(12, 1.2, n),
        'Budget_USD': np.random.lognormal(8, 0.6, n),
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
        data['Weight_kg'].append(data['Height_cm'][i] * 0.7 + np.random.normal(0, 8))
        data['BMI'].append(data['Weight_kg'][i] / ((data['Height_cm'][i] / 100) ** 2))
        
        # Experience correlates with age but capped
        max_exp = max(0, data['Age_years'][i] - 22)
        actual_exp = min(max_exp, np.random.randint(0, max_exp + 5) if max_exp > 0 else 0)
        data['Experience_years'].append(actual_exp)
        
        # Income correlates with education, experience, and performance
        income_factor = (
            data['Education_years'][i] * 0.15 +
            actual_exp * 0.08 +
            data['Performance_Rating'][i] * 0.2 +
            np.random.normal(8.5, 0.7)
        )
        data['Income_USD'].append(np.exp(income_factor))
        
        # Savings correlate with income and age
        savings_rate = 0.1 + (data['Age_years'][i] - 25) * 0.002
        savings_rate = max(0.05, min(0.3, savings_rate))
        data['Savings_USD'].append(data['Income_USD'][i] * savings_rate * np.random.uniform(0.5, 1.5))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Apply department-based adjustments
    dept_mapping = {
        'Engineering': {'avg_income_mult': 1.2, 'avg_education': 16},
        'Marketing': {'avg_income_mult': 1.0, 'avg_education': 14},
        'Sales': {'avg_income_mult': 1.1, 'avg_education': 13},
        'HR': {'avg_income_mult': 0.9, 'avg_education': 15},
        'Finance': {'avg_income_mult': 1.15, 'avg_education': 16}
    }
    
    for dept, props in dept_mapping.items():
        mask = df['Department'] == dept
        df.loc[mask, 'Income_USD'] *= props['avg_income_mult']
    
    return df

def main():
    # Configure page first
    configure_page()
    
    # Add custom CSS
    add_custom_css()
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Advanced BI Dashboard - Dynamic Data Visualization</h1>', unsafe_allow_html=True)
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🔧 Dashboard Controls</div>', unsafe_allow_html=True)
        
        # File upload section
        st.markdown("### 📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Choose a spreadsheet file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your data file (CSV or Excel format)"
        )
        
        # Initialize session state safely
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'file_uploaded' not in st.session_state:
            st.session_state.file_uploaded = False
        
        # Load data when file is uploaded
        if uploaded_file is not None:
            with st.spinner("Loading data..."):
                st.session_state.data = load_data(uploaded_file)
            
            if st.session_state.data is not None:
                st.success(f"✅ Data loaded successfully!")
                st.info(f"Shape: {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
    
    # Main content area
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Display data summary
        display_data_summary(df)
        
        # Get numeric columns for visualization
        numeric_columns = get_numeric_columns(df)
        all_columns = df.columns.tolist()
        
        if len(numeric_columns) < 2:
            st.error("⚠️ Dataset needs at least 2 numeric columns for scatter plot visualization.")
            st.info("Please upload a dataset with more numeric columns.")
            return
        
        # Advanced Configuration Section
        st.markdown("### 🎯 Advanced Visualization Controls")
        
        # Main control columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_axis = st.selectbox(
                "📊 Select X-Axis",
                options=numeric_columns,
                index=0,
                help="Choose the column for X-axis"
            )
        
        with col2:
            y_axis = st.selectbox(
                "📊 Select Y-Axis",
                options=numeric_columns,
                index=1 if len(numeric_columns) > 1 else 0,
                help="Choose the column for Y-axis"
            )
        
        with col3:
            color_options = ["None"] + all_columns
            color_column = st.selectbox(
                "🎨 Color Mapping",
                options=color_options,
                index=0,
                help="Choose a column for color grouping"
            )
        
        with col4:
            size_options = ["None"] + numeric_columns
            size_column = st.selectbox(
                "📏 Size Mapping",
                options=size_options,
                index=0,
                help="Choose a column for bubble size"
            )
        
        # Advanced Options
        st.markdown("### ⚙️ Advanced Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_transform = st.checkbox(
                "🔄 Z-score X-axis",
                value=False,
                help="Apply Z-score transformation to X-axis"
            )
        
        with col2:
            y_transform = st.checkbox(
                "🔄 Z-score Y-axis", 
                value=False,
                help="Apply Z-score transformation to Y-axis"
            )
        
        with col3:
            show_outliers = st.checkbox(
                "⚠️ Highlight Outliers",
                value=True,
                help="Highlight outliers with special markers"
            )
        
        with col4:
            show_trend = st.checkbox(
                "📈 Show Trend Line",
                value=True,
                help="Add linear regression trend line"
            )
        
        # Additional controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_marginals = st.checkbox(
                "📊 Marginal Distributions",
                value=False,
                help="Show marginal histograms"
            )
        
        with col2:
            bubble_size_max = st.slider(
                "📏 Max Bubble Size",
                min_value=10,
                max_value=100,
                value=60,
                step=5,
                help="Maximum size for bubble markers"
            )
        
        with col3:
            st.markdown("##### 🎨 Plot Style")
            plot_template = st.selectbox(
                "Template",
                options=['plotly_white', 'plotly', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white'],
                index=0,
                help="Choose plot template"
            )
        
        # Create and display the enhanced scatter plot
        st.markdown("### 📈 Advanced Scatter Plot Analysis")
        
        if x_axis and y_axis:
            with st.spinner("Creating advanced visualization..."):
                fig, stats_results = create_enhanced_scatter_plot(
                    df, x_axis, y_axis, 
                    color_var=color_column if color_column != "None" else None,
                    size_var=size_column if size_column != "None" else None,
                    x_transform=x_transform,
                    y_transform=y_transform,
                    show_outliers=show_outliers,
                    show_trend=show_trend,
                    show_marginals=show_marginals,
                    bubble_size_max=bubble_size_max
                )
                
                if fig is not None:
                    # Update template
                    fig.update_layout(template=plot_template)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display comprehensive statistics
                    if stats_results:
                        display_comprehensive_statistics(stats_results, x_axis, y_axis)
                        
                        # Quick insights
                        with st.expander("💡 Quick Insights", expanded=True):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                correlation = stats_results['correlation']
                                if abs(correlation) > 0.7:
                                    insight = "🔥 Strong relationship detected!"
                                    color = "green"
                                elif abs(correlation) > 0.3:
                                    insight = "📊 Moderate relationship found."
                                    color = "orange"
                                else:
                                    insight = "📉 Weak relationship observed."
                                    color = "red"
                                st.markdown(f"<p style='color: {color};'><b>{insight}</b></p>", unsafe_allow_html=True)
                                st.metric("Correlation", f"{correlation:.4f}")
                            
                            with col2:
                                r_squared = stats_results['r_squared']
                                st.metric("R-squared", f"{r_squared:.4f}")
                                st.caption(f"{r_squared*100:.1f}% of variance explained")
                            
                            with col3:
                                p_value = stats_results['p_value']
                                significance = "Significant" if p_value < 0.05 else "Not Significant"
                                st.metric("P-value", f"{p_value:.2e}")
                                st.caption(f"Statistical significance: {significance}")
                        
                        # Export functionality
                        st.markdown("### 💾 Export Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("📊 Export Plot Data"):
                                plot_cols = [x_axis, y_axis]
                                if color_column != "None":
                                    plot_cols.append(color_column)
                                if size_column != "None":
                                    plot_cols.append(size_column)
                                
                                export_df = df[plot_cols].copy()
                                
                                # Add transformed columns if applied
                                if x_transform:
                                    analyzer = AdvancedScatterPlotAnalyzer()
                                    export_df[f'{x_axis}_zscore'] = analyzer.calculate_z_score(df[x_axis])
                                if y_transform:
                                    analyzer = AdvancedScatterPlotAnalyzer()
                                    export_df[f'{y_axis}_zscore'] = analyzer.calculate_z_score(df[y_axis])
                                
                                csv = export_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Plot Data CSV",
                                    data=csv,
                                    file_name=f'plot_data_{x_axis}_vs_{y_axis}.csv',
                                    mime='text/csv'
                                )
                        
                        with col2:
                            if st.button("📈 Export Statistics"):
                                stats_text = f"""Advanced Scatter Plot Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Variables: {y_axis} vs {x_axis}
Sample Size: {stats_results['n_points']:,} points

CORRELATION ANALYSIS:
- Pearson Correlation: {stats_results['correlation']:.6f}
- Spearman Correlation: {stats_results['spearman_corr']:.6f}
- Kendall's Tau: {stats_results['kendall_corr']:.6f}
- R-squared: {stats_results['r_squared']:.6f}

REGRESSION ANALYSIS:
- Slope: {stats_results['slope']:.6f}
- Intercept: {stats_results['intercept']:.6f}
- Standard Error: {stats_results['std_error']:.6f}
- P-value: {stats_results['p_value']:.2e}

TRANSFORMATIONS APPLIED:
- X-axis Z-score: {x_transform}
- Y-axis Z-score: {y_transform}
- Outlier Detection: {show_outliers}
"""
                                
                                st.download_button(
                                    label="📥 Download Statistics Report",
                                    data=stats_text,
                                    file_name=f'statistics_{x_axis}_vs_{y_axis}.txt',
                                    mime='text/plain'
                                )
                        
                        with col3:
                            if st.button("🔍 Export Full Analysis"):
                                # Comprehensive export including outlier detection
                                analyzer = AdvancedScatterPlotAnalyzer()
                                analyzer.df = df
                                
                                export_df = df.copy()
                                
                                # Add Z-scores
                                export_df[f'{x_axis}_zscore'] = analyzer.calculate_z_score(df[x_axis])
                                export_df[f'{y_axis}_zscore'] = analyzer.calculate_z_score(df[y_axis])
                                
                                # Add outlier flags
                                x_z = analyzer.calculate_z_score(df[x_axis])
                                y_z = analyzer.calculate_z_score(df[y_axis])
                                export_df['is_outlier'] = (np.abs(x_z) > 2) | (np.abs(y_z) > 2)
                                
                                csv = export_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Analysis CSV",
                                    data=csv,
                                    file_name=f'full_analysis_{x_axis}_vs_{y_axis}.csv',
                                    mime='text/csv'
                                )
        
        # Data preview and quality section
        with st.expander("🔍 Data Preview & Quality", expanded=False):
            st.markdown("### Raw Data Preview")
            st.dataframe(df.head(20), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Data Quality Summary")
                quality_df = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null %': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(quality_df, use_container_width=True)
            
            with col2:
                st.markdown("### 🔢 Numeric Columns Summary")
                if len(numeric_columns) > 0:
                    st.dataframe(df[numeric_columns].describe(), use_container_width=True)
                else:
                    st.info("No numeric columns found.")
    
    else:
        # Welcome message when no data is loaded
        st.markdown("""
        <div class="upload-section">
            <h3>👋 Welcome to the Advanced BI Dashboard!</h3>
            <p>This powerful data visualization tool offers:</p>
            <ul>
                <li>📁 Support for CSV and Excel files</li>
                <li>🔄 Z-score transformations for data standardization</li>
                <li>📊 Advanced scatter plots with variable sizing</li>
                <li>⚠️ Automatic outlier detection and highlighting</li>
                <li>📈 Comprehensive statistical analysis</li>
                <li>🎨 Color mapping and interactive visualizations</li>
                <li>💾 Data export capabilities</li>
            </ul>
            <p><strong>👈 Use the sidebar to upload your data file and get started!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data option
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎲 Generate Sample Dataset", help="Create a comprehensive sample dataset to explore features"):
                with st.spinner("Generating comprehensive sample data..."):
                    sample_data = create_sample_dataset()
                    st.session_state.data = sample_data
                    st.success("✅ Sample dataset created successfully!")
                    st.rerun()
        
        with col2:
            st.markdown("""
            ### 💡 Sample Analysis Ideas:
            - **Career Development**: Experience vs Income by Department
            - **Physical Health**: Height vs Weight by Gender 
            - **Performance Analysis**: Education vs Performance Rating
            - **Economic Patterns**: Income vs Savings by Education Level
            """)
        
        # Feature showcase
        st.markdown("### ✨ Key Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
            #### 🔄 Data Transformations
            - **Z-score Standardization**: Normalize variables to same scale
            - **Outlier Detection**: Automatic identification using statistical methods
            - **Variable Sizing**: Map third dimension to bubble size
            """)
        
        with feature_col2:
            st.markdown("""
            #### 📊 Advanced Analytics
            - **Multiple Correlation Measures**: Pearson, Spearman, Kendall
            - **Regression Analysis**: Linear trend lines with R²
            - **Statistical Significance**: P-values and confidence intervals
            """)
        
        with feature_col3:
            st.markdown("""
            #### 🎨 Visualization Options
            - **Color Mapping**: Group by categorical variables
            - **Interactive Plots**: Hover details and zoom capabilities  
            - **Marginal Distributions**: Optional histogram overlays
            """)

if __name__ == "__main__":
    main()