import React, { useState, useEffect } from 'react';
import BirdMap from './components/BirdMap';
import './App.css';

function App() {
  const [geoJson, setGeoJson] = useState(null);
  const [allPredictions, setAllPredictions] = useState(null);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedBird, setSelectedBird] = useState('Goleag'); // 'Goleag' or 'Whteag'
  const [availableDates, setAvailableDates] = useState([]);

  useEffect(() => {
    // Fetch GeoJSON
    // In Vite, public assets are at the root path
    fetch('sweden.geo.json')
      .then(res => res.json())
      .then(data => setGeoJson(data))
      .catch(err => console.error("Error loading GeoJSON", err));

    // Fetch Predictions
    fetch('predictions.json')
      .then(res => res.json())
      .then(data => {
        setAllPredictions(data);
        const dates = Object.keys(data).sort();
        setAvailableDates(dates);
        if (dates.length > 0) {
          // Select today or the first available date
          setSelectedDate(dates[0]);
        }
      })
      .catch(err => console.error("Error loading predictions", err));
  }, []);

  const currentPredictions = allPredictions && selectedDate && allPredictions[selectedDate] && allPredictions[selectedDate][selectedBird]
    ? allPredictions[selectedDate][selectedBird]
    : {};

  return (
    <div className="App">
      <div className="sidebar">
        <h1>BirdUp Sweden</h1>
        
        <div className="control-group">
          <label>Bird Type</label>
          <div className="toggle-group">
            <button 
              className={selectedBird === 'Goleag' ? 'active' : ''} 
              onClick={() => setSelectedBird('Goleag')}>
              Golden Eagle
            </button>
            <button 
              className={selectedBird === 'Whteag' ? 'active' : ''} 
              onClick={() => setSelectedBird('Whteag')}>
              White-tailed Eagle
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>Date: {selectedDate}</label>
          {availableDates.length > 0 && (
            <input 
              type="range" 
              min="0" 
              max={availableDates.length - 1} 
              value={availableDates.indexOf(selectedDate) !== -1 ? availableDates.indexOf(selectedDate) : 0}
              onChange={(e) => setSelectedDate(availableDates[e.target.value])}
            />
          )}
        </div>
        
        <div className="info">
          <p>Visualizing sighting probabilities for the coming week.</p>
          <p>Data updated daily via Hopsworks Feature Store.</p>
        </div>
      </div>
      
      <div className="map-container">
        <BirdMap geoJson={geoJson} predictions={currentPredictions} />
      </div>
    </div>
  );
}

export default App;