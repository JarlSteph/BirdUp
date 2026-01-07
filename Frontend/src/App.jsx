import React, { useState, useEffect } from 'react';
import BirdMap from './components/BirdMap';
import './App.css';

// Update paths to be relative
const goldenSound = new Audio('sounds/golden.mp3');
const whiteSound = new Audio('sounds/white.mp3');


function App() {
  const [geoJson, setGeoJson] = useState(null);
  const [allPredictions, setAllPredictions] = useState(null);
  const [performanceData, setPerformanceData] = useState(null);
  const [selectedDate, setSelectedDate] = useState('');
  const [selectedBird, setSelectedBird] = useState('goleag'); // 'goleag' or 'whteag'
  const [availableDates, setAvailableDates] = useState([]);
  const [isMuted, setIsMuted] = useState(false);
  const [showStats, setShowStats] = useState(false);

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

    // Fetch Performance Data
    fetch('performance.json')
      .then(res => res.json())
      .then(data => setPerformanceData(data))
      .catch(err => console.error("Error loading performance stats", err));
  }, []);

  const currentPredictions = allPredictions && selectedDate && allPredictions[selectedDate] && allPredictions[selectedDate][selectedBird]
    ? allPredictions[selectedDate][selectedBird]
    : {};

  const calculateMetrics = (cm) => {
    // cm = [[tn, fp], [fn, tp]]
    const tn = cm[0][0];
    const fp = cm[0][1];
    const fn = cm[1][0];
    const tp = cm[1][1];
    
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const balancedAcc = (recall + specificity) / 2;
    
    return { precision, recall, f1, balancedAcc, tn, fp, fn, tp };
  };

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
          <p><strong>BirdUp</strong> forecasts daily sighting probabilities for Golden and White-tailed Eagles across Sweden's 25 provinces.</p>
          <p>Our PyTorch neural networks analyze historical patterns and daily weather metrics to predict bird presence for the coming week.</p>
          <p>Data pipeline and models are managed via the <strong>Hopsworks Feature Store</strong>, with daily inference updates.</p>
          <p>Data powered by <a href="https://ebird.org/home">Ebird</a></p>
        </div>
      </div>
      
      {/* Right-side Stats Panel */}
      <div className={`stats-sidebar ${showStats ? 'open' : ''}`}>
        <button className="close-stats" onClick={() => setShowStats(false)}>×</button>
        <h2>Live Hindcast Performance</h2>
        <p className="stats-intro">Real-time confusion matrix based on recent predictions vs actual observations.</p>
        
        {performanceData ? (
          Object.keys(performanceData).map(bird => {
            const data = performanceData[bird];
            const metrics = calculateMetrics(data.cm);
            const birdName = bird === 'goleag' ? 'Golden Eagle' : 'White-tailed Eagle';
            
            return (
              <div key={bird} className="bird-stats">
                <h3>{birdName}</h3>
                <div className="meta-info">
                  <span>Samples: {data.total_samples}</span>
                  <span>Updated: {data.last_updated}</span>
                </div>
                
                <div className="confusion-matrix">
                  <div className="cm-header"></div>
                  <div className="cm-header">Pred 0</div>
                  <div className="cm-header">Pred 1</div>
                  
                  <div className="cm-label">Actual 0</div>
                  <div className="cm-cell tn" title="True Negative">{metrics.tn}</div>
                  <div className="cm-cell fp" title="False Positive">{metrics.fp}</div>
                  
                  <div className="cm-label">Actual 1</div>
                  <div className="cm-cell fn" title="False Negative">{metrics.fn}</div>
                  <div className="cm-cell tp" title="True Positive">{metrics.tp}</div>
                </div>

                <div className="metrics-grid" style={{ justifyContent: 'space-around' }}>
                  <div className="metric">
                    <span className="label">Bal. Accuracy</span>
                    <span className="value">{metrics.balancedAcc.toFixed(3)}</span>
                  </div>
                  <div className="metric">
                    <span className="label">F1 Score</span>
                    <span className="value">{metrics.f1.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            );
          })
        ) : (
          <p>Loading stats...</p>
        )}
      </div>

      <button className="stats-toggle" onClick={() => setShowStats(!showStats)}>
        📊 HINDCAST
      </button>

      <div className="map-container">
        <BirdMap geoJson={geoJson} predictions={currentPredictions} />
      </div>
    </div>
  );
}

export default App;