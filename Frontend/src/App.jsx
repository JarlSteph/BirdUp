import React, { useState, useEffect } from 'react';
import BirdMap from './components/BirdMap';
import './App.css';

// Update paths to be relative
const goldenSound = new Audio('sounds/golden.mp3');
const whiteSound = new Audio('sounds/white.mp3');


function App() {
  const [geoJson, setGeoJson] = useState(null);
  const [allPredictions, setAllPredictions] = useState(null);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedBird, setSelectedBird] = useState('goleag'); // 'goleag' or 'whteag'
  const [availableDates, setAvailableDates] = useState([]);
  const [isMuted, setIsMuted] = useState(false);

  // Instantly set volume to 0 when isMuted changes
  useEffect(() => {
    goldenSound.volume = isMuted ? 0 : 1;
    whiteSound.volume = isMuted ? 0 : 1;
  }, [isMuted]);

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
        console.log("Predictions loaded:", Object.keys(data).length, "days.");
        setAllPredictions(data);
        const dates = Object.keys(data).sort();
        setAvailableDates(dates);
        if (dates.length > 0) {
          console.log("Setting initial date to:", dates[0]);
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
      <div className="sidebar dangerous">
        <div className="header-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1>BirdUp Sweden</h1>
          <button 
            className="mute-toggle"
            onClick={() => setIsMuted(!isMuted)}
            style={{ 
              background: '#222', 
              border: '1px solid #d40000', 
              borderRadius: '4px',
              padding: '8px 12px',
              cursor: 'pointer', 
              fontSize: '1.2rem',
              boxShadow: isMuted ? 'none' : '0 0 10px rgba(212, 0, 0, 0.3)',
              transition: 'all 0.2s ease'
            }}
          >
            {isMuted ? '🔇 MUTED' : '🔊 SOUND'}
          </button>
        </div>
        
        <div className="control-group">
          <label>SELECT PREDATOR</label>
          <div className="toggle-group">
            <button 
              className={selectedBird === 'goleag' ? 'active' : ''} 
              onClick={() => {
                setSelectedBird('goleag');
                whiteSound.pause();
                whiteSound.currentTime = 0;
                goldenSound.currentTime = 0;
                goldenSound.play(); // Play sound; volume is handled by useEffect
              }}>
              <img src="imgs/goldeag.png" alt="Golden Eagle" className="bird-img" />
              Golden Eagle
            </button>
            <button 
              className={selectedBird === 'whteag' ? 'active' : ''} 
              onClick={() => {
                setSelectedBird('whteag');
                goldenSound.pause();
                goldenSound.currentTime = 0;
                whiteSound.currentTime = 0;
                whiteSound.play(); // Play sound; volume is handled by useEffect
              }}>
              <img src="imgs/whiteeag.png" alt="White-tailed Eagle" className="bird-img" />
              White-tailed Eagle
            </button>
          </div>
        </div>

        <div className="control-group">
          <label>SIGHTING WINDOW: {selectedDate}</label>
          {availableDates.length > 0 && (
            <input 
              type="range" 
              min="0" 
              max={availableDates.length - 1} 
              value={availableDates.indexOf(selectedDate) !== -1 ? availableDates.indexOf(selectedDate) : 0}
              onChange={(e) => {
                setSelectedDate(availableDates[e.target.value]);
              }}
              style={{ accentColor: '#d40000' }}
            />
          )}
        </div>
        
        <div className="info">
          <p>Visualizing lethal sighting probabilities for the coming week.</p>
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