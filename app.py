from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import yfinance as yf
from datetime import datetime, timedelta
import json

import os
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    ticker = data.get('ticker', 'AAPL')
    days = data.get('days', 30)
    contamination = data.get('contamination', 0.1)
    
    try:
        # Fetch data
        if ticker == "SAMPLE":
            np.random.seed(42)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            prices = 100 + np.cumsum(np.random.randn(30) * 2)
            prices[10] = 85
            prices[15] = 125
            prices[22] = 140
            df = pd.DataFrame({
                'Date': dates,
                'Close': prices
            })
        else:
            stock_data = yf.download(ticker, period=f"{days}d", progress=False)
            df = pd.DataFrame(stock_data)
            df = df.reset_index()
            df = df[['Date', 'Close']]
        
        # Calculate returns
        df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # Anomaly detection
        X = df[['Daily_Return']].fillna(0).values
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df['Anomaly'] = iso_forest.fit_predict(X)
        df['Is_Anomaly'] = (df['Anomaly'] == -1).astype(int)
        df['Anomaly_Score'] = iso_forest.score_samples(X)
        
        # Prepare response
        anomalies = df[df['Is_Anomaly'] == 1]
        
        response = {
            'success': True,
            'ticker': ticker,
            'total_days': len(df),
            'anomalies_count': int(df['Is_Anomaly'].sum()),
            'avg_return': float(df['Daily_Return'].mean()),
            'volatility': float(df['Daily_Return'].std()),
            'price_min': float(df['Close'].min()),
            'price_max': float(df['Close'].max()),
            'chart_data': {
                'dates': df['Date'].astype(str).tolist(),
                'prices': df['Close'].round(2).tolist(),
                'returns': df['Daily_Return'].round(3).tolist(),
                'anomalies': df['Is_Anomaly'].tolist()
            },
            'anomaly_details': anomalies[['Date', 'Close', 'Daily_Return']].to_dict('records')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
