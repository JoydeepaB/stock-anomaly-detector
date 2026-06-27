from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import yfinance as yf
from datetime import datetime
import json

app = Flask(__name__)

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Anomaly Detector</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
* {margin: 0; padding: 0; box-sizing: border-box;}
body {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px;}
.container {max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3); padding: 40px;}
header {text-align: center; margin-bottom: 40px;}
header h1 {font-size: 2.5em; color: #2c3e50; margin-bottom: 10px;}
header p {font-size: 1.1em; color: #7f8c8d;}
.controls {display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;}
.input-group {display: flex; flex-direction: column; gap: 8px;}
.input-group label {font-weight: 600; color: #2c3e50; font-size: 0.95em;}
.input-group input {padding: 10px; border: 2px solid #ecf0f1; border-radius: 6px; font-size: 1em; transition: border-color 0.3s;}
.input-group input:focus {outline: none; border-color: #667eea;}
.input-group span {color: #667eea; font-weight: 600;}
button {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 6px; font-size: 1em; font-weight: 600; cursor: pointer; transition: transform 0.2s; grid-column: 1 / -1;}
button:hover {transform: translateY(-2px);}
.metrics {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;}
.metric-card {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 8px; text-align: center;}
.metric-label {display: block; color: #7f8c8d; font-size: 0.9em; margin-bottom: 8px;}
.metric-value {display: block; font-size: 1.8em; font-weight: 700; color: #2c3e50;}
.loading {display: flex; justify-content: center; align-items: center; gap: 10px; padding: 40px; font-size: 1.2em; color: #667eea;}
.loading::after {content: ''; width: 20px; height: 20px; border: 3px solid #667eea; border-top-color: transparent; border-radius: 50%; animation: spin 0.8s linear infinite;}
@keyframes spin {to {transform: rotate(360deg);}}
.chart-container, .returns-container {margin: 40px 0; padding: 20px; background: #f8f9fa; border-radius: 8px;}
.chart-container h2, .returns-container h2 {margin-bottom: 20px; color: #2c3e50;}
.anomalies-container {margin: 40px 0; padding: 20px; background: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107;}
.anomalies-container h2 {margin-bottom: 20px; color: #856404;}
table {width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden;}
table th {background: #667eea; color: white; padding: 12px; text-align: left; font-weight: 600;}
table td {padding: 12px; border-bottom: 1px solid #ecf0f1;}
table tr:hover {background: #f8f9fa;}
.error {padding: 20px; background: #f8d7da; color: #721c24; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #f5c6cb;}
@media (max-width: 768px) {.container {padding: 20px;} header h1 {font-size: 1.8em;} .controls {grid-template-columns: 1fr;}}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Stock Anomaly Detector</h1>
            <p>Detect unusual price movements using Machine Learning</p>
        </header>

        <div class="controls">
            <div class="input-group">
                <label>Stock Ticker:</label>
                <input type="text" id="ticker" value="AAPL" placeholder="e.g., INFY, TCS, AAPL">
            </div>
            <div class="input-group">
                <label>Days of History:</label>
                <input type="range" id="days" min="10" max="365" value="30">
                <span id="daysValue">30</span>
            </div>
            <div class="input-group">
                <label>Anomaly Sensitivity (%):</label>
                <input type="range" id="contamination" min="5" max="30" value="10" step="1">
                <span id="contaminationValue">10</span>
            </div>
            <button id="analyzeBtn">Analyze</button>
        </div>

        <div class="metrics" id="metrics">
            <div class="metric-card">
                <span class="metric-label">Total Days</span>
                <span class="metric-value" id="totalDays">-</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Anomalies</span>
                <span class="metric-value" id="anomaliesCount">-</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Avg Return</span>
                <span class="metric-value" id="avgReturn">-</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Volatility</span>
                <span class="metric-value" id="volatility">-</span>
            </div>
        </div>

        <div class="loading" id="loading" style="display: none;">
            <span>Analyzing...</span>
        </div>

        <div class="chart-container" id="chartContainer" style="display: none;">
            <h2>Stock Price Analysis</h2>
            <canvas id="priceChart"></canvas>
        </div>

        <div class="returns-container" id="returnsContainer" style="display: none;">
            <h2>Daily Returns</h2>
            <canvas id="returnsChart"></canvas>
        </div>

        <div class="anomalies-container" id="anomaliesContainer" style="display: none;">
            <h2>🚨 Detected Anomalies</h2>
            <table id="anomaliesTable">
                <thead><tr><th>Date</th><th>Price ($)</th><th>Return (%)</th></tr></thead>
                <tbody id="anomaliesTableBody"></tbody>
            </table>
        </div>

        <div class="error" id="error" style="display: none;"></div>
    </div>

    <script>
const ticker = document.getElementById('ticker');
const days = document.getElementById('days');
const daysValue = document.getElementById('daysValue');
const contamination = document.getElementById('contamination');
const contaminationValue = document.getElementById('contaminationValue');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const error = document.getElementById('error');

let priceChart = null;
let returnsChart = null;

days.addEventListener('input', (e) => daysValue.textContent = e.target.value);
contamination.addEventListener('input', (e) => contaminationValue.textContent = e.target.value);

analyzeBtn.addEventListener('click', async () => {
    const tickerValue = ticker.value.toUpperCase() || 'AAPL';
    const daysValue = parseInt(days.value);
    const contaminationValue = parseFloat(contamination.value) / 100;

    loading.style.display = 'flex';
    error.style.display = 'none';

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: tickerValue,
                days: daysValue,
                contamination: contaminationValue
            })
        });

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Analysis failed');
        }

        document.getElementById('totalDays').textContent = data.total_days;
        document.getElementById('anomaliesCount').textContent = data.anomalies_count;
        document.getElementById('avgReturn').textContent = data.avg_return.toFixed(2) + '%';
        document.getElementById('volatility').textContent = data.volatility.toFixed(2) + '%';

        const priceCtx = document.getElementById('priceChart').getContext('2d');
        if (priceChart) priceChart.destroy();
        
        priceChart = new Chart(priceCtx, {
            type: 'line',
            data: {
                labels: data.chart_data.dates,
                datasets: [{
                    label: 'Price',
                    data: data.chart_data.prices,
                    borderColor: '#3498db',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 4,
                    pointBackgroundColor: data.chart_data.anomalies.map(a => a ? '#e74c3c' : '#3498db')
                }]
            },
            options: {responsive: true, plugins: {legend: {display: true}}}
        });

        const returnsCtx = document.getElementById('returnsChart').getContext('2d');
        if (returnsChart) returnsChart.destroy();
        
        returnsChart = new Chart(returnsCtx, {
            type: 'bar',
            data: {
                labels: data.chart_data.dates,
                datasets: [{
                    label: 'Daily Return (%)',
                    data: data.chart_data.returns,
                    backgroundColor: data.chart_data.anomalies.map(a => a ? '#e74c3c' : '#2ecc71')
                }]
            },
            options: {responsive: true, plugins: {legend: {display: false}}}
        });

        const anomaliesTableBody = document.getElementById('anomaliesTableBody');
        anomaliesTableBody.innerHTML = '';
        if (data.anomaly_details.length > 0) {
            data.anomaly_details.forEach(anomaly => {
                const row = `<tr><td>${new Date(anomaly.Date).toLocaleDateString()}</td><td>$${anomaly.Close.toFixed(2)}</td><td>${anomaly.Daily_Return.toFixed(3)}%</td></tr>`;
                anomaliesTableBody.innerHTML += row;
            });
            document.getElementById('anomaliesContainer').style.display = 'block';
        } else {
            document.getElementById('anomaliesContainer').style.display = 'none';
        }

        document.getElementById('chartContainer').style.display = 'block';
        document.getElementById('returnsContainer').style.display = 'block';
    } catch (err) {
        error.textContent = 'Error: ' + err.message;
        error.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
});

analyzeBtn.click();
    </script>
</body>
</html>"""

@app.route('/')
def index():
    return HTML_CONTENT

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    ticker = data.get('ticker', 'AAPL')
    days = data.get('days', 30)
    contamination = data.get('contamination', 0.1)
    
    try:
        if ticker == "SAMPLE":
            np.random.seed(42)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            prices = 100 + np.cumsum(np.random.randn(30) * 2)
            prices[10] = 85
            prices[15] = 125
            prices[22] = 140
            df = pd.DataFrame({'Date': dates, 'Close': prices})
        else:
            try:
                stock_data = yf.download(ticker, period=f"{days}d", progress=False)
                if len(stock_data) == 0:
                    raise Exception("No data")
                df = pd.DataFrame(stock_data)
                df = df.reset_index()
                df = df[['Date', 'Close']]
            except:
                np.random.seed(42)
                dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                prices = 100 + np.cumsum(np.random.randn(30) * 2)
                prices[10] = 85
                prices[15] = 125
                prices[22] = 140
                df = pd.DataFrame({'Date': dates, 'Close': prices})
        
        df['Daily_Return'] = df['Close'].pct_change() * 100
        df['Daily_Return'] = df['Daily_Return'].fillna(0)
        
        X = df[['Daily_Return']].values
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        df['Anomaly'] = iso_forest.fit_predict(X)
        df['Is_Anomaly'] = (df['Anomaly'] == -1).astype(int)
        anomalies = df[df['Is_Anomaly'] == 1]
        
        avg_return = float(df['Daily_Return'].mean())
        volatility = float(df['Daily_Return'].std())
        
        if np.isnan(avg_return):
            avg_return = 0
        if np.isnan(volatility):
            volatility = 0
        
        response = {
            'success': True,
            'ticker': ticker,
            'total_days': int(len(df)),
            'anomalies_count': int(df['Is_Anomaly'].sum()),
            'avg_return': avg_return,
            'volatility': volatility,
            'price_min': float(df['Close'].min()),
            'price_max': float(df['Close'].max()),
            'chart_data': {
                'dates': [str(d.date()) for d in df['Date']],
                'prices': [float(x) for x in df['Close'].round(2)],
                'returns': [float(x) for x in df['Daily_Return'].round(3)],
                'anomalies': [int(x) for x in df['Is_Anomaly']]
            },
            'anomaly_details': [
                {
                    'Date': str(row['Date'].date()),
                    'Close': float(row['Close']),
                    'Daily_Return': float(row['Daily_Return'])
                }
                for _, row in anomalies.iterrows()
            ]
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False)
