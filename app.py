from flask import Flask, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)
EXCEL_FILE = 'uploads/US_Disease_Insurance.xlsx'

# Ensure the Excel file exists
if not os.path.exists(EXCEL_FILE):
    raise FileNotFoundError(f"Excel file {EXCEL_FILE} not found")

# Read the Excel file once at startup
df = pd.read_excel(EXCEL_FILE)

# Clean the data
def clean_data(df):
    # Replace suppressed data symbols with NaN
    suppressed_symbols = ['~', '*', '****', '#']
    df['DataValue'] = df['DataValue'].replace(suppressed_symbols, float('nan'))
    df['LowConfidenceLimit'] = df['LowConfidenceLimit'].replace(suppressed_symbols, float('nan'))
    df['HighConfidenceLimit'] = df['HighConfidenceLimit'].replace(suppressed_symbols, float('nan'))
    
    # Convert DataValue, LowConfidenceLimit, and HighConfidenceLimit to numeric
    df['DataValue'] = pd.to_numeric(df['DataValue'], errors='coerce')
    df['LowConfidenceLimit'] = pd.to_numeric(df['LowConfidenceLimit'], errors='coerce')
    df['HighConfidenceLimit'] = pd.to_numeric(df['HighConfidenceLimit'], errors='coerce')
    
    # Drop rows with NaN in DataValue or LowConfidenceLimit
    df = df.dropna(subset=['DataValue', 'LowConfidenceLimit'])
    return df

@app.route('/')
def index():
    # Get unique topics for the dropdown
    topics = df['Topic'].unique().tolist()
    return render_template('index.html', topics=topics)

@app.route('/get_data/<topic>')
def get_data(topic):
    try:
        # Clean the data
        cleaned_df = clean_data(df)
        
        # Filter by selected topic
        filtered_df = cleaned_df[cleaned_df['Topic'] == topic]
        
        if filtered_df.empty:
            return jsonify({'error': f'No valid data for topic {topic}'}), 400
        
        # Prepare data for scatter plot
        data = {
            'x': filtered_df['DataValue'].tolist(),
            'y': filtered_df['LowConfidenceLimit'].tolist(),
            'labels': filtered_df['LocationDesc'].tolist(),
            'stratifications': filtered_df['Stratification1'].tolist(),
            'x_label': 'Data Value',
            'y_label': 'Low Confidence Limit'
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)