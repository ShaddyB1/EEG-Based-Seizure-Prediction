import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalAveragePooling1D, LSTM, BatchNormalization
import warnings
import joblib
from datetime import datetime
import pywt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import threading
import queue
from scipy.stats import mode
import numpy.fft as fft
import time
warnings.filterwarnings('ignore')

class AdvancedEEGProcessor:
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.models = {}
        self.scaler = StandardScaler()
        self.wavelet = 'db4'
        self.feature_buffer = queue.Queue(maxsize=1000)
        self.prediction_buffer = queue.Queue(maxsize=1000)
        self.visualization_buffer = []
        self.is_processing = False
        self.current_position = 0
        plt.style.use('default')

        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        
    def generate_synthetic_data(self, duration=60):
        
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        
        background = (
            0.5 * np.sin(2 * np.pi * 10 * t) +  # Alpha
            0.2 * np.sin(2 * np.pi * 20 * t) +  # Beta
            0.1 * np.sin(2 * np.pi * 5 * t) +   # Theta
            0.1 * np.random.normal(0, 1, len(t)) # Noise
        )
        
        
        seizure_patterns = np.zeros_like(t)
        labels = np.zeros_like(t)
        
        # seizure pattern categories
        seizure_periods = [
            (10, 15),   # Regular seizure
            (25, 30),   # Complex seizure
            (40, 45),   # High-frequency seizure
            (50, 55)    # Mixed pattern
        ]
        
        for start, end in seizure_periods:
            start_idx = int(start * self.sampling_rate)
            end_idx = int(end * self.sampling_rate)
            t_seizure = t[start_idx:end_idx]
            
            
            if start == 10:  
                pattern = (
                    2.0 * np.sin(2 * np.pi * 3 * t_seizure) +
                    1.5 * np.sin(2 * np.pi * 15 * t_seizure)
                )
            elif start == 25:  
                pattern = (
                    2.5 * np.sin(2 * np.pi * 4 * t_seizure) *
                    np.exp(-0.1 * t_seizure) +
                    np.random.normal(0, 0.5, len(t_seizure))
                )
            elif start == 40:  
                pattern = (
                    3.0 * np.sin(2 * np.pi * 20 * t_seizure) +
                    1.0 * np.sin(2 * np.pi * 40 * t_seizure)
                )
            else:  
                pattern = (
                    2.0 * np.sin(2 * np.pi * 3 * t_seizure) +
                    1.5 * np.sin(2 * np.pi * 15 * t_seizure) +
                    1.0 * np.sin(2 * np.pi * 30 * t_seizure) *
                    np.exp(-0.2 * t_seizure)
                )
            
            seizure_patterns[start_idx:end_idx] = pattern
            labels[start_idx:end_idx] = 1
            
        
        eeg_data = background + seizure_patterns
        
        
        artifacts = self.generate_artifacts(t)
        eeg_data += artifacts
        
        return eeg_data, labels
    
    def generate_artifacts(self, t):
        """Generate common EEG artifacts."""
        artifacts = np.zeros_like(t)
        
        blink_times = np.random.choice(len(t)-100, 5)
        for bt in blink_times:
            artifacts[bt:bt+100] += 2 * np.exp(-np.linspace(0, 2, 100)**2)
        
        
        muscle_times = np.random.choice(len(t)-500, 3)
        for mt in muscle_times:
            artifacts[mt:mt+500] += 0.5 * np.random.normal(0, 1, 500)
        
        
        movement_times = np.random.choice(len(t)-1000, 2)
        for mvt in movement_times:
            artifacts[mvt:mvt+1000] += 1.5 * np.sin(2 * np.pi * 0.5 * np.linspace(0, 2, 1000))
        
        return artifacts
    
    def extract_advanced_features(self, segment):
        """Extract comprehensive set of EEG features."""
        features = {}
        
        
        features.update({
            'mean': np.mean(segment),
            'std': np.std(segment),
            'skewness': float(pd.Series(segment).skew()),
            'kurtosis': float(pd.Series(segment).kurtosis()),
            'rms': np.sqrt(np.mean(np.square(segment))),
            'zero_crossings': np.sum(np.diff(np.signbit(segment).astype(int))),
            'line_length': np.sum(np.abs(np.diff(segment))),
            'mobility': self._hjorth_mobility(segment),
            'complexity': self._hjorth_complexity(segment),
            'energy': np.sum(np.square(segment)),
            'nonlinear_energy': self._nonlinear_energy(segment)
        })
        
        
        freqs, psd = signal.welch(segment, self.sampling_rate)
        
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45),
            'high_gamma': (45, 100),
            'ripple': (100, 200),
            'fast_ripple': (200, 500)
        }
        
        total_power = np.sum(psd)
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(psd[mask])
            features[f'{band_name}_power'] = band_power
            features[f'{band_name}_ratio'] = band_power / total_power if total_power > 0 else 0
        
        
        coeffs = pywt.wavedec(segment, self.wavelet, level=4)
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_l{i}_energy'] = np.sum(coeff**2)
            features[f'wavelet_l{i}_entropy'] = self._entropy(coeff)
        
        return features
    
    def _hjorth_mobility(self, data):
        
        diff = np.diff(data)
        return np.std(diff) / np.std(data)
    
    def _hjorth_complexity(self, data):
        
        diff1 = np.diff(data)
        diff2 = np.diff(diff1)
        return (np.std(diff2) * np.std(data)) / (np.std(diff1) ** 2)
    
    def _nonlinear_energy(self, data):
       
        return np.sum(data[1:-1]**2 - data[:-2] * data[2:])
    
    def _entropy(self, data):
        
        data = np.abs(data)
        data_norm = data / np.sum(data)
        return -np.sum(data_norm * np.log2(data_norm + 1e-10))
    
    def train_multiple_models(self, X, y):
        
        X_scaled = self.scaler.fit_transform(X)
        
        
        models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42)
        }
        
        print("Training classical models...")
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            self.models[name] = model
        
        
        if len(X) > 1000:
            print("Training deep learning models...")
            X_reshaped = X_scaled.reshape(-1, X_scaled.shape[1], 1)
            deep_models = self.create_deep_models((X_scaled.shape[1], 1))
            
            for name, model in deep_models.items():
                print(f"Training {name}...")
                model.fit(X_reshaped, y, epochs=50, batch_size=32, 
                         validation_split=0.2, verbose=0)
                self.models[name] = model
        
        
        print("Creating ensemble...")
        self.models['ensemble'] = VotingClassifier(
            estimators=[(name, model) for name, model in models.items()
                       if name not in ['cnn', 'lstm']],
            voting='soft'
        )
        self.models['ensemble'].fit(X_scaled, y)
    
    def evaluate_all_models(self, X_test, y_test):
        
        X_scaled = self.scaler.transform(X_test)
        results = {}
        
        for name, model in self.models.items():
            if name in ['cnn', 'lstm']:
                X_reshaped = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                y_pred = (model.predict(X_reshaped) > 0.5).astype(int)
            else:
                y_pred = model.predict(X_scaled)
            
            results[name] = {
                'report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'predictions': y_pred
            }
        
        return results

    def prepare_data(self, eeg_data, labels, window_size=1):
       
        window_points = int(window_size * self.sampling_rate)
        features = []
        window_labels = []
        
        for i in range(0, len(eeg_data) - window_points, window_points//2):  
            segment = eeg_data[i:i + window_points]
            segment_features = self.extract_advanced_features(segment)
            features.append(segment_features)
            
            
            window_label = mode(labels[i:i + window_points])[0]
            window_labels.append(window_label)
        
        return pd.DataFrame(features), np.array(window_labels)
    
    def create_deep_models(self, input_shape):
        
        cnn = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(64, 3, activation='relu'),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            Conv1D(128, 3, activation='relu'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Advanced LSTM
        lstm = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return {'cnn': cnn, 'lstm': lstm}

    def start_real_time_processing(self):
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_real_time)
        self.processing_thread.start()

    def _process_real_time(self):
        """Process EEG data in real-time."""
        window = []
        while self.is_processing:
            if not self.feature_buffer.empty():
                sample = self.feature_buffer.get()
                window.append(sample)
                
                
                try:
                    self.visualization_buffer.append(sample)
                    if len(self.visualization_buffer) > self.sampling_rate * 10:  
                        self.visualization_buffer.pop(0)
                except AttributeError:
                    self.visualization_buffer = [sample]
                
                if len(window) >= self.sampling_rate:
                    features = self.extract_advanced_features(np.array(window))
                    X = pd.DataFrame([features])
                    X_scaled = self.scaler.transform(X)
                    
                    predictions = {}
                    for name, model in self.models.items():
                        if name in ['cnn', 'lstm']:
                            X_reshaped = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                            pred = model.predict(X_reshaped, verbose=0)[0][0]
                        else:
                            pred = model.predict_proba(X_scaled)[0][1]
                        predictions[name] = pred
                    
                    self.prediction_buffer.put((predictions, np.mean(list(predictions.values()))))
                    window = window[self.sampling_rate//2:]  

    def stop_real_time_processing(self):
        
        self.is_processing = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()

    

    def create_dashboard(self):
        
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Real-time EEG Seizure Detection Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
            
            html.Div([
                html.Div([
                    html.Button('Start Processing', 
                              id='start-button',
                              n_clicks=0,
                              style={'marginRight': '10px',
                                    'backgroundColor': '#2ecc71',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'borderRadius': '5px'}),
                    html.Button('Stop Processing',
                              id='stop-button',
                              n_clicks=0,
                              style={'backgroundColor': '#e74c3c',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '10px 20px',
                                    'borderRadius': '5px'}),
                    html.Div(id='processing-status',
                            style={'marginTop': '10px',
                                  'padding': '10px',
                                  'backgroundColor': '#f8f9fa',
                                  'borderRadius': '5px'})
                ], style={'textAlign': 'center', 'marginBottom': '20px'}),
                
                html.Div([
                    dcc.Graph(id='live-eeg-graph'),
                    dcc.Graph(id='live-prediction-graph'),
                    dcc.Graph(id='frequency-analysis-graph'),
                    dcc.Graph(id='model-comparison-graph')
                ]),
                
                dcc.Interval(
                    id='interval-component',
                    interval=100,  # update every 100 ms
                    n_intervals=0
                )
            ])
        ], style={'padding': '20px'})
        
        @app.callback(
            [Output('live-eeg-graph', 'figure'),
             Output('live-prediction-graph', 'figure'),
             Output('frequency-analysis-graph', 'figure'),
             Output('model-comparison-graph', 'figure')],
            Input('interval-component', 'n_intervals')
        )
        def update_graphs(n):
            
            eeg_data = self.visualization_buffer[-1000:] if hasattr(self, 'visualization_buffer') else []
            predictions = list(self.prediction_buffer.queue)
            
            
            eeg_fig = {
                'data': [{
                    'y': eeg_data,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'EEG Signal',
                    'line': {'color': '#2980b9', 'width': 2}
                }],
                'layout': {
                    'title': 'Live EEG Signal',
                    'xaxis': {'title': 'Samples', 'showgrid': True},
                    'yaxis': {'title': 'Amplitude', 'showgrid': True},
                    'height': 400,
                    'template': 'plotly_white',
                    'uirevision': 'constant'  
                }
            }
            
            
            pred_data = [p[1] for p in predictions[-100:]] if predictions else []
            pred_fig = {
                'data': [{
                    'y': pred_data,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Seizure Probability',
                    'line': {'color': '#27ae60', 'width': 2}
                }],
                'layout': {
                    'title': 'Seizure Probability',
                    'xaxis': {'title': 'Windows', 'showgrid': True},
                    'yaxis': {'title': 'Probability', 'range': [0, 1], 'showgrid': True},
                    'height': 400,
                    'template': 'plotly_white',
                    'uirevision': 'constant',
                    'shapes': [{
                        'type': 'line',
                        'y0': 0.5,
                        'y1': 0.5,
                        'x0': 0,
                        'x1': len(pred_data),
                        'line': {'color': '#c0392b', 'dash': 'dash'}
                    }]
                }
            }
            
            
            if len(eeg_data) >= self.sampling_rate:
                freqs, psd = signal.welch(eeg_data[-self.sampling_rate:], fs=self.sampling_rate)
                freq_fig = {
                    'data': [{
                        'x': freqs,
                        'y': psd,
                        'type': 'scatter',
                        'mode': 'lines',
                        'name': 'Power Spectrum',
                        'line': {'color': '#8e44ad', 'width': 2}
                    }],
                    'layout': {
                        'title': 'Frequency Analysis',
                        'xaxis': {'title': 'Frequency (Hz)', 'showgrid': True},
                        'yaxis': {'title': 'Power', 'showgrid': True},
                        'height': 400,
                        'template': 'plotly_white',
                        'uirevision': 'constant'
                    }
                }
            else:
                freq_fig = {'data': [], 'layout': {'title': 'Frequency Analysis'}}
            
            
            if predictions:
                latest_preds = predictions[-1][0]
                model_fig = {
                    'data': [{
                        'x': list(latest_preds.keys()),
                        'y': list(latest_preds.values()),
                        'type': 'bar',
                        'marker': {'color': ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']}
                    }],
                    'layout': {
                        'title': 'Model Predictions',
                        'xaxis': {'title': 'Models', 'showgrid': True},
                        'yaxis': {'title': 'Probability', 'range': [0, 1], 'showgrid': True},
                        'height': 400,
                        'template': 'plotly_white',
                        'uirevision': 'constant'
                    }
                }
            else:
                model_fig = {'data': [], 'layout': {'title': 'Model Predictions'}}
            
            return eeg_fig, pred_fig, freq_fig, model_fig
        
        @app.callback(
            Output('processing-status', 'children'),
            [Input('start-button', 'n_clicks'),
             Input('stop-button', 'n_clicks')]
        )
        def control_processing(start_clicks, stop_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return "Click 'Start Processing' to begin"
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'start-button':
                self.start_real_time_processing()
                return "Processing Started"
            else:
                self.stop_real_time_processing()
                return "Processing Stopped"
        
        return app

    def run_dashboard(self, host='localhost', port=8050):
        
        app = self.create_dashboard()
        app.run_server(host=host, port=port, debug=False)

def main():
    
    processor = AdvancedEEGProcessor()
    
    try:
        
        print("Generating synthetic data...")
        eeg_data, labels = processor.generate_synthetic_data(duration=60)
        
        
        print("Preparing data...")
        X, y = processor.prepare_data(eeg_data, labels)
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        
        print("Training models...")
        processor.train_multiple_models(X_train, y_train)
        
        
        print("Starting dashboard...")
        print("Navigate to http://localhost:8050 to view the dashboard")
        
        
        def feed_data():
            while True:  # Add continuous loop
                try:
                    if processor.is_processing:
                        if processor.current_position < len(eeg_data):
                            processor.feature_buffer.put(eeg_data[processor.current_position])
                            processor.current_position += 1
                        else:
                            # Reset position to create continuous loop
                            processor.current_position = 0
                        time.sleep(1/processor.sampling_rate)
                except Exception as e:
                    print(f"Error in feed_data: {str(e)}")
        
        
        data_thread = threading.Thread(target=feed_data)
        data_thread.daemon = True
        data_thread.start()
        
        
        processor.run_dashboard()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()