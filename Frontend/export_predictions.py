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
    try:
        project = hopsworks.login(project="BirdUp", api_key_value=os.getenv("HOPSWORKS_API_KEY"))
        fs = project.get_feature_store()
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # ---------------------------------------------------------
    # 1. Fetch Data
    # ---------------------------------------------------------
    print("Fetching feature groups...")
    try:
        # Predictions
        pred_fg = fs.get_feature_group(name="bird_sighting_predictions", version=1)
        # Select specific columns to improve read stability
        pred_cols = ["observation_date", "region", "bird_type", "probability", "prediction"]
        try:
            pred_df = pred_fg.select(pred_cols).read()
        except:
             print("Arrow Flight (preds) failed, trying Hive...")
             pred_df = pred_fg.select(pred_cols).read(read_options={"use_hive": True})
        
        # Actuals
        actual_fg = fs.get_feature_group(name="birding", version=1)
        actual_cols = ["observation_date", "region", "bird_type", "observation_count"]
        try:
            actual_df = actual_fg.select(actual_cols).read()
        except:
            print("Arrow Flight (actuals) failed, trying Hive...")
            actual_df = actual_fg.select(actual_cols).read(read_options={"use_hive": True})

        # Standardize Dates
        pred_df["observation_date"] = pd.to_datetime(pred_df["observation_date"])
        actual_df["observation_date"] = pd.to_datetime(actual_df["observation_date"])
        
        # Standardize Columns for Merge
        pred_df["obs_date_str"] = pred_df["observation_date"].dt.strftime("%Y-%m-%d")
        actual_df["obs_date_str"] = actual_df["observation_date"].dt.strftime("%Y-%m-%d")
        
        # Standardize Strings
        pred_df["region"] = pred_df["region"].astype(str)
        actual_df["region"] = actual_df["region"].astype(str)
        pred_df["bird_type"] = pred_df["bird_type"].astype(str)
        actual_df["bird_type"] = actual_df["bird_type"].astype(str)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # ---------------------------------------------------------
    # 2. Process Future Predictions (predictions.json)
    # ---------------------------------------------------------
    today = datetime.now()
    today_str = today.strftime("%Y-%m-%d")
    
    df_future = pred_df[pred_df['obs_date_str'] >= today_str].copy()
    
    export_data = {}
    print(f"Processing {len(df_future)} future predictions...")
    
    for _, row in df_future.iterrows():
        date = row['obs_date_str']
        region = row['region']
        bird = row['bird_type']
        prob = row['probability']
        
        if date not in export_data:
            export_data[date] = {}
        if bird not in export_data[date]:
            export_data[date][bird] = {}
        
        export_data[date][bird][region] = prob

    output_dir = os.path.join(os.path.dirname(__file__), 'public')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_path = os.path.join(output_dir, 'predictions.json')
    print(f"Writing predictions to {pred_path}...")
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    # ---------------------------------------------------------
    # 3. Calculate Live Performance (performance.json)
    # ---------------------------------------------------------
    print("Calculating live performance (Confusion Matrix)...")
    
    # Merge
    merged_df = pd.merge(
        pred_df, 
        actual_df, 
        on=["obs_date_str", "region", "bird_type"], 
        how="inner",
        suffixes=("_pred", "_actual")
    )
    
    print(f"Found {len(merged_df)} overlapping records for validation.")
    
    performance_data = {}
    today_disp = today.strftime("%Y-%m-%d")

    for bird in ["goleag", "whteag"]:
        subset = merged_df[merged_df["bird_type"] == bird]
        
        if subset.empty:
            cm = [[0, 0], [0, 0]]
        else:
            # Actual Binary
            y_true = (subset["observation_count"] > 0).astype(int)
            
            # Predicted Binary
            # Use 'prediction' column if exists (from inference_daily), else threshold probability
            if "prediction" in subset.columns:
                y_pred = subset["prediction"].astype(int)
            else:
                y_pred = (subset["probability"] >= 0.5).astype(int)
                
            # Compute CM
            # sklearn confusion_matrix returns [[TN, FP], [FN, TP]]
            # We implement manually to be safe/simple without sklearn dep if not needed, 
            # but simple logic is fine.
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            
            cm = [[int(tn), int(fp)], [int(fn), int(tp)]]
            
        performance_data[bird] = {
            "cm": cm,
            "last_updated": today_disp,
            "total_samples": int(subset.shape[0]) if not subset.empty else 0
        }

    perf_path = os.path.join(output_dir, 'performance.json')
    print(f"Writing performance stats to {perf_path}...")
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump(performance_data, f, ensure_ascii=False, indent=2)

    print("Done.")

if __name__ == "__main__":
    export_predictions()
