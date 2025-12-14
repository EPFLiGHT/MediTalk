"""
HTML report generator for Whisper benchmark results.
Creates interactive visualizations and comparison tables.
"""

import json
from pathlib import Path
from typing import Dict
from datetime import datetime


def generate_html_report(results: Dict, output_dir: Path) -> str:
    """
    Generate an HTML report with benchmark results.
    
    Args:
        results: Dictionary with benchmark results for all models
        output_dir: Directory to save the report
        
    Returns:
        Path to generated HTML file
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper ASR Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 50px;
        }}
        
        .section h2 {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        
        .metric-card h3 {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        
        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-card .unit {{
            font-size: 0.9em;
            color: #888;
            margin-left: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .chart {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }}
        
        .best {{
            color: #10b981;
            font-weight: bold;
        }}
        
        .worst {{
            color: #ef4444;
            font-weight: bold;
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎤 Whisper ASR Benchmark Report</h1>
            <p>Medical Speech Recognition Performance Analysis</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="content">
            {_generate_summary_section(results)}
            {_generate_comparison_table(results)}
            {_generate_charts(results)}
            {_generate_detailed_metrics(results)}
        </div>
        
        <div class="footer">
            <p>MediTalk Whisper Benchmark • Generated automatically from benchmark results</p>
        </div>
    </div>
    
    <script>
        {_generate_chart_scripts(results)}
    </script>
</body>
</html>
"""
    
    output_file = output_dir / "benchmark_report.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return str(output_file)


def _generate_summary_section(results: Dict) -> str:
    """Generate summary metrics section."""
    if not results:
        return "<p>No results available</p>"
    
    # Find best model based on WER
    best_model = min(results.items(), key=lambda x: x[1].get('wer', {}).get('mean', 1.0))
    best_model_name = best_model[0]
    best_wer = best_model[1].get('wer', {}).get('mean', 0)
    
    # Calculate average metrics across all models
    avg_wer = sum(r.get('wer', {}).get('mean', 0) for r in results.values()) / len(results)
    total_samples = list(results.values())[0].get('sample_count', 0)
    
    html = f"""
    <div class="section">
        <h2>📊 Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Best Model</h3>
                <div class="value">{best_model_name}</div>
            </div>
            <div class="metric-card">
                <h3>Best WER</h3>
                <div class="value">{best_wer:.4f}<span class="unit"></span></div>
            </div>
            <div class="metric-card">
                <h3>Average WER</h3>
                <div class="value">{avg_wer:.4f}<span class="unit"></span></div>
            </div>
            <div class="metric-card">
                <h3>Samples Tested</h3>
                <div class="value">{total_samples}<span class="unit">files</span></div>
            </div>
        </div>
    </div>
    """
    return html


def _generate_comparison_table(results: Dict) -> str:
    """Generate comparison table."""
    html = """
    <div class="section">
        <h2>📈 Model Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>WER (mean)</th>
                    <th>CER (mean)</th>
                    <th>Latency P95 (s)</th>
                    <th>RTF (mean)</th>
                    <th>Samples</th>
                    <th>Errors</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Find best/worst for highlighting
    if results:
        best_wer = min(r.get('wer', {}).get('mean', 1.0) for r in results.values())
        worst_wer = max(r.get('wer', {}).get('mean', 0) for r in results.values())
    
    for model, data in results.items():
        wer_mean = data.get('wer', {}).get('mean', 0)
        cer_mean = data.get('cer', {}).get('mean', 0)
        lat_p95 = data.get('latency', {}).get('p95', 0)
        rtf_mean = data.get('rtf', {}).get('mean', 0)
        samples = data.get('sample_count', 0)
        errors = data.get('error_count', 0)
        
        wer_class = 'best' if wer_mean == best_wer else ('worst' if wer_mean == worst_wer else '')
        
        html += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td class="{wer_class}">{wer_mean:.4f}</td>
                    <td>{cer_mean:.4f}</td>
                    <td>{lat_p95:.3f}</td>
                    <td>{rtf_mean:.3f}</td>
                    <td>{samples}</td>
                    <td>{errors}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    return html


def _generate_charts(results: Dict) -> str:
    """Generate chart placeholders."""
    html = """
    <div class="section">
        <h2>📉 Performance Visualizations</h2>
        
        <div class="chart" id="wer-chart"></div>
        <div class="chart" id="latency-chart"></div>
        <div class="chart" id="rtf-chart"></div>
    </div>
    """
    return html


def _generate_detailed_metrics(results: Dict) -> str:
    """Generate detailed metrics for each model."""
    html = """
    <div class="section">
        <h2>🔍 Detailed Metrics</h2>
    """
    
    for model, data in results.items():
        wer = data.get('wer', {})
        cer = data.get('cer', {})
        latency = data.get('latency', {})
        rtf = data.get('rtf', {})
        
        html += f"""
        <div style="margin-bottom: 40px;">
            <h3 style="color: #764ba2; font-size: 1.5em; margin-bottom: 15px;">{model.upper()}</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>WER Mean</h3>
                    <div class="value">{wer.get('mean', 0):.4f}</div>
                </div>
                <div class="metric-card">
                    <h3>WER Median</h3>
                    <div class="value">{wer.get('median', 0):.4f}</div>
                </div>
                <div class="metric-card">
                    <h3>CER Mean</h3>
                    <div class="value">{cer.get('mean', 0):.4f}</div>
                </div>
                <div class="metric-card">
                    <h3>Latency P95</h3>
                    <div class="value">{latency.get('p95', 0):.2f}<span class="unit">s</span></div>
                </div>
                <div class="metric-card">
                    <h3>RTF Mean</h3>
                    <div class="value">{rtf.get('mean', 0):.3f}x</div>
                </div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html


def _generate_chart_scripts(results: Dict) -> str:
    """Generate Plotly chart scripts."""
    # Prepare data for charts
    models = list(results.keys())
    wer_means = [results[m].get('wer', {}).get('mean', 0) for m in models]
    cer_means = [results[m].get('cer', {}).get('mean', 0) for m in models]
    lat_p95s = [results[m].get('latency', {}).get('p95', 0) for m in models]
    rtf_means = [results[m].get('rtf', {}).get('mean', 0) for m in models]
    
    script = f"""
    // WER Comparison Chart
    var werData = [{{
        x: {models},
        y: {wer_means},
        type: 'bar',
        marker: {{
            color: '#667eea',
            line: {{
                color: '#764ba2',
                width: 2
            }}
        }},
        text: {[f'{w:.4f}' for w in wer_means]},
        textposition: 'outside',
    }}];
    
    var werLayout = {{
        title: 'Word Error Rate (WER) by Model - Lower is Better',
        xaxis: {{ title: 'Model Size' }},
        yaxis: {{ title: 'WER' }},
        showlegend: false,
        height: 400
    }};
    
    Plotly.newPlot('wer-chart', werData, werLayout);
    
    // Latency Chart
    var latencyData = [{{
        x: {models},
        y: {lat_p95s},
        type: 'bar',
        marker: {{
            color: '#10b981',
            line: {{
                color: '#059669',
                width: 2
            }}
        }},
        text: {[f'{l:.2f}s' for l in lat_p95s]},
        textposition: 'outside',
    }}];
    
    var latencyLayout = {{
        title: 'Latency P95 by Model - Lower is Better',
        xaxis: {{ title: 'Model Size' }},
        yaxis: {{ title: 'Latency (seconds)' }},
        showlegend: false,
        height: 400
    }};
    
    Plotly.newPlot('latency-chart', latencyData, latencyLayout);
    
    // RTF Chart
    var rtfData = [{{
        x: {models},
        y: {rtf_means},
        type: 'bar',
        marker: {{
            color: '#f59e0b',
            line: {{
                color: '#d97706',
                width: 2
            }}
        }},
        text: {[f'{r:.3f}x' for r in rtf_means]},
        textposition: 'outside',
    }}];
    
    var rtfLayout = {{
        title: 'Real-Time Factor (RTF) by Model - Lower is Better (< 1.0 = faster than real-time)',
        xaxis: {{ title: 'Model Size' }},
        yaxis: {{ title: 'RTF' }},
        showlegend: false,
        height: 400,
        shapes: [{{
            type: 'line',
            x0: -0.5,
            x1: {len(models) - 0.5},
            y0: 1.0,
            y1: 1.0,
            line: {{
                color: 'red',
                width: 2,
                dash: 'dash'
            }}
        }}],
        annotations: [{{
            x: {len(models) / 2},
            y: 1.0,
            text: 'Real-time threshold',
            showarrow: false,
            yshift: 10
        }}]
    }};
    
    Plotly.newPlot('rtf-chart', rtfData, rtfLayout);
    """
    
    return script
