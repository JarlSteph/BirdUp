import json

def check_keys():
    try:
        with open('Frontend/public/predictions.json', 'r', encoding='utf-8') as f:
            preds = json.load(f)
        
        with open('Frontend/public/sweden.geo.json', 'r', encoding='utf-8') as f:
            geo = json.load(f)
            
        geo_regions = set()
        for feature in geo['features']:
            geo_regions.add(feature['properties']['landskap'])
            
        print(f"Found {len(geo_regions)} regions in GeoJSON.")
        
        # Get one date/bird entry to check keys
        first_date = list(preds.keys())[0]
        first_bird = list(preds[first_date].keys())[0]
        pred_regions = set(preds[first_date][first_bird].keys())
        
        print(f"Found {len(pred_regions)} regions in Predictions.")
        
        missing_in_preds = geo_regions - pred_regions
        missing_in_geo = pred_regions - geo_regions
        
        if not missing_in_preds and not missing_in_geo:
            print("SUCCESS: All regions match!")
        else:
            print("MISMATCH FOUND!")
            if missing_in_preds:
                print(f"Regions in GeoJSON but NOT in Predictions: {missing_in_preds}")
            if missing_in_geo:
                print(f"Regions in Predictions but NOT in GeoJSON: {missing_in_geo}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_keys()
