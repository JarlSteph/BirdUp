import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import { scaleSequential } from 'd3-scale';
import { interpolateReds } from 'd3-scale-chromatic'; // Changed to Reds for "dangerous" look
import 'leaflet/dist/leaflet.css';

const BirdMap = ({ geoJson, predictions }) => {
  // Use a deep red/dark scale
  const colorScale = scaleSequential(interpolateReds).domain([0, 1]);

  const style = (feature) => {
    const regionName = feature.properties.landskap;
    const probability = predictions ? predictions[regionName] : undefined;
    
    const p = probability !== undefined ? probability : 0;
    
    return {
      fillColor: probability !== undefined ? colorScale(p) : '#1a1a1a', // Darker empty state
      weight: 1.5, // Thicker borders
      opacity: 1,
      color: '#444', // Darker borders instead of white
      dashArray: '', // Solid lines for a cleaner look
      fillOpacity: 0.8
    };
  };

  const onEachFeature = (feature, layer) => {
    const regionName = feature.properties.landskap;
    const probability = predictions ? predictions[regionName] : undefined;
    const text = probability !== undefined ? `${(probability * 100).toFixed(1)}%` : 'No Data';
    
    // Custom tooltip styling (requires CSS)
    layer.bindTooltip(
        `<div class="dangerous-tooltip"><strong>${regionName}</strong><br/>Threat Level: ${text}</div>`,
        { sticky: true, className: 'custom-tooltip' }
    );
  };

  return (
    <MapContainer 
      center={[62.0, 15.0]} 
      zoom={5} 
      style={{ height: '100%', width: '100%', background: '#0a0a0a' }}
    >
      {/* Dark Matter tiles for an epic/dangerous feel */}
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />
      {geoJson && (
        <GeoJSON 
          data={geoJson} 
          style={style} 
          onEachFeature={onEachFeature} 
          key={JSON.stringify(predictions)} 
        /> 
      )}
    </MapContainer>
  );
};

export default BirdMap;