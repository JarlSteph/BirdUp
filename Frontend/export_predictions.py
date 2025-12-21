import os
import json
import pandas as pd
import hopsworks
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def export_predictions():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(project="BirdUp", api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    fs = project.get_feature_store()
    
    print("Fetching predictions feature group...")
    try:
        fg = fs.get_feature_group(name="bird_sighting_predictions", version=1)
        df = fg.read()
    except Exception as e:
        print(f"Error fetching feature group: {e}")
        return

    # Filter for future dates (today onwards)
    today = datetime.now().strftime("%Y-%m-%d")
    # specific format from the notebook/script might be YYYY-MM-DD or datetime object
    # Let's inspect the data types if we could, but assuming standard string or timestamp
    
    # Ensure observation_date is string YYYY-MM-DD for consistency
    if pd.api.types.is_datetime64_any_dtype(df['observation_date']):
        df['observation_date'] = df['observation_date'].dt.strftime('%Y-%m-%d')
    else:
        # It might be in a different format, but let's assume it's ISO or convertible
        df['observation_date'] = pd.to_datetime(df['observation_date']).dt.strftime('%Y-%m-%d')

    # Filter
    df_future = df[df['observation_date'] >= today].copy()
    
    # Structure: { date: { bird_type: { region: probability } } }
    export_data = {}
    
    print("Formatting data...")
    for _, row in df_future.iterrows():
        date = row['observation_date']
        region = row['region']
        bird = row['bird_type']
        prob = row['probability']
        
        if date not in export_data:
            export_data[date] = {}
        if bird not in export_data[date]:
            export_data[date][bird] = {}
        
        export_data[date][bird][region] = prob

    # Output path
    output_dir = os.path.join(os.path.dirname(__file__), 'public')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predictions.json')
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    export_predictions()
