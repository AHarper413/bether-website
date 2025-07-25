import os
import sys
import json
import pandas as pd
from datetime import datetime, date
import subprocess
import glob

def run_mlb_model():
    """Run the main MLB model and return the predictions"""
    print("🚀 Running MLB betting model...")
    
    try:
        # Run your main model file (now in same directory)
        result = subprocess.run([sys.executable, 'mlbtargets.py'], 
                              capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"❌ Model failed with error: {result.stderr}")
            return None
            
        print("✅ Model completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("⏰ Model timed out after 30 minutes")
        return None
    except Exception as e:
        print(f"❌ Error running model: {e}")
        return None

def find_latest_predictions_file():
    """Find the most recent predictions CSV file"""
    today_str = date.today().strftime("%Y_%m_%d")
    current_year = datetime.now().year
    
    # Look for today's file first in the output directory
    expected_filename = f"./mlb_improved_data_{current_year}/mlb_predictions_v5_{today_str}.csv"
    
    if os.path.exists(expected_filename):
        return expected_filename
    
    # Look for any recent predictions file
    pattern = f"./mlb_improved_data_{current_year}/mlb_predictions_v5_*.csv"
    files = glob.glob(pattern)
    
    if files:
        # Return the most recent file
        return max(files, key=os.path.getmtime)
    
    # If no files found, return None
    print("❌ No predictions file found")
    return None

# [Rest of the functions remain the same as before - create_picks_json, create_picks_html, etc.]

def create_picks_json(predictions_df):
    """Convert predictions to JSON format for website"""
    today = date.today().strftime("%B %d, %Y")
    
    # Count signals
    strong_home_count = len(predictions_df[predictions_df['strong_home_favorite'] == 'YES'])
    strong_away_count = len(predictions_df[predictions_df['strong_away_favorite'] == 'YES']) 
    total_signals = strong_home_count + strong_away_count
    
    over_8_5_count = len(predictions_df[predictions_df['lean_over_8_5'] == 'YES'])
    f5_signals_count = len(predictions_df[predictions_df['f5_over_signal'] == 'YES'])
    
    # Get best bets (highest confidence)
    best_ml_bets = []
    for _, game in predictions_df.iterrows():
        home_prob = game['home_win_probability']
        away_prob = game['away_win_probability']
        
        if home_prob > 0.58:  # Strong home favorite
            confidence = "HIGH" if home_prob > 0.62 else "MEDIUM"
            best_ml_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": f"{game['home_team']} ML",
                "probability": f"{home_prob:.1%}",
                "confidence": confidence,
                "reasoning": f"Model favors home team with {home_prob:.1%} win probability"
            })
        elif away_prob > 0.58:  # Strong away favorite  
            confidence = "HIGH" if away_prob > 0.62 else "MEDIUM"
            best_ml_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": f"{game['away_team']} ML", 
                "probability": f"{away_prob:.1%}",
                "confidence": confidence,
                "reasoning": f"Model favors away team with {away_prob:.1%} win probability"
            })
    
    # Get total bets
    total_bets = []
    for _, game in predictions_df.iterrows():
        if game['lean_over_8_5'] == 'YES':
            prob = game['over_8_5_probability']
            confidence = "HIGH" if prob > 0.60 else "MEDIUM"
            total_bets.append({
                "game": f"{game['away_team']} @ {game['home_team']}",
                "pick": "Over 8.5 Runs",
                "probability": f"{prob:.1%}",
                "confidence": confidence,
                "predicted_total": game['predicted_total_runs']
            })
    
    # Create JSON structure
    picks_data = {
        "date": today,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S EST"),
        "model_version": "v5.0_XGBoost",
        "summary": {
            "total_games": len(predictions_df),
            "moneyline_signals": total_signals,
            "over_8_5_signals": over_8_5_count,
            "f5_signals": f5_signals_count
        },
        "best_moneyline_bets": best_ml_bets[:5],  # Top 5
        "best_total_bets": total_bets[:3],  # Top 3
        "season_record": {"wins": 156, "losses": 100, "percentage": 61.0}  # Update this manually
    }
    
    return picks_data

def create_picks_html(picks_data):
    """Create HTML page for today's picks"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MLB Model Picks - {picks_data['date']} | BetHer</title>
    <meta name="description" content="Today's MLB betting picks from our advanced machine learning model. 61% accuracy this season.">
    <style>
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.6;
            color: #1e293b;
            background: #f8fafc;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            border-bottom: 2px solid #e11d48;
            padding-bottom: 1rem;
        }}
        .header h1 {{
            color: #e11d48;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }}
        .record-banner {{
            background: linear-gradient(135deg, #e11d48 0%, #8b5cf6 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 2rem;
        }}
        .record-banner h2 {{
            margin: 0;
            font-size: 1.5rem;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #e11d48;
        }}
        .picks-section {{
            margin-bottom: 2rem;
        }}
        .picks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
        }}
        .pick-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
        }}
        .pick-card.high {{
            border-left: 4px solid #10b981;
        }}
        .pick-card.medium {{
            border-left: 4px solid #f59e0b;
        }}
        .confidence-badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
            text-transform: uppercase;
        }}
        .confidence-high {{
            background: #d1fae5;
            color: #065f46;
        }}
        .confidence-medium {{
            background: #fef3c7;
            color: #92400e;
        }}
        .back-link {{
            text-align: center;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e2e8f0;
        }}
        .back-link a {{
            color: #e11d48;
            text-decoration: none;
            font-weight: 600;
        }}
        @media (max-width: 768px) {{
            .container {{
                padding: 1rem;
            }}
            .header h1 {{
                font-size: 2rem;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MLB Model Picks</h1>
            <h2>{picks_data['date']}</h2>
            <p>Last Updated: {picks_data['last_updated']}</p>
        </div>

        <div class="record-banner">
            <h2>Season Record: {picks_data['season_record']['wins']}-{picks_data['season_record']['losses']} ({picks_data['season_record']['percentage']:.1f}%)</h2>
            <p>Advanced XGBoost Model with Feature Selection</p>
        </div>

        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['total_games']}</div>
                <div>Games Today</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['moneyline_signals']}</div>
                <div>ML Signals</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['over_8_5_signals']}</div>
                <div>Over 8.5 Signals</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{picks_data['summary']['f5_signals']}</div>
                <div>F5 Signals</div>
            </div>
        </div>
"""

    # Add best moneyline bets
    if picks_data['best_moneyline_bets']:
        html_content += """
        <div class="picks-section">
            <h3>🎯 Best Moneyline Bets</h3>
            <div class="picks-grid">
"""
        for bet in picks_data['best_moneyline_bets']:
            confidence_class = bet['confidence'].lower()
            html_content += f"""
                <div class="pick-card {confidence_class}">
                    <h4>{bet['game']}</h4>
                    <p><strong>{bet['pick']}</strong></p>
                    <p>Probability: {bet['probability']}</p>
                    <span class="confidence-badge confidence-{confidence_class}">{bet['confidence']}</span>
                    <p><small>{bet['reasoning']}</small></p>
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    # Add total bets
    if picks_data['best_total_bets']:
        html_content += """
        <div class="picks-section">
            <h3>📊 Best Total Bets</h3>
            <div class="picks-grid">
"""
        for bet in picks_data['best_total_bets']:
            confidence_class = bet['confidence'].lower()
            html_content += f"""
                <div class="pick-card {confidence_class}">
                    <h4>{bet['game']}</h4>
                    <p><strong>{bet['pick']}</strong></p>
                    <p>Probability: {bet['probability']}</p>
                    <p>Predicted Total: {bet['predicted_total']:.1f}</p>
                    <span class="confidence-badge confidence-{confidence_class}">{bet['confidence']}</span>
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    html_content += """
        <div class="back-link">
            <p><strong>Disclaimer:</strong> These are model predictions for educational purposes. Always bet responsibly.</p>
            <p><a href="/index.html">← Back to BetHer Home</a></p>
        </div>
    </div>
</body>
</html>
"""
    
    return html_content

def main():
    print("🚀 Starting MLB picks automation...")
    
    # Run the model
    model_success = run_mlb_model()
    if not model_success:
        print("❌ Model execution failed, cannot generate picks")
        return
    
    # Find the predictions file
    predictions_file = find_latest_predictions_file()
    if not predictions_file:
        print("❌ No predictions file found")
        return
    
    print(f"📊 Found predictions file: {predictions_file}")
    
    # Load predictions
    try:
        predictions_df = pd.read_csv(predictions_file)
        print(f"✅ Loaded {len(predictions_df)} game predictions")
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return
    
    # Create JSON data
    picks_data = create_picks_json(predictions_df)
    
    # Save JSON file
    with open('daily_picks.json', 'w') as f:
        json.dump(picks_data, f, indent=2)
    print("✅ Created daily_picks.json")
    
    # Create HTML page
    html_content = create_picks_html(picks_data)
    with open('mlb-picks-today.html', 'w') as f:
        f.write(html_content)
    print("✅ Created mlb-picks-today.html")
    
    print(f"🎯 Summary for {picks_data['date']}:")
    print(f"   📊 {picks_data['summary']['total_games']} games analyzed")
    print(f"   🎯 {picks_data['summary']['moneyline_signals']} ML signals")
    print(f"   📈 {picks_data['summary']['over_8_5_signals']} total signals")
    print(f"   ⚾ {picks_data['summary']['f5_signals']} F5 signals")

if __name__ == "__main__":
    main()