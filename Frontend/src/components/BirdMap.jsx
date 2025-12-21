import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import { scaleSequential } from 'd3-scale';
import { interpolateYlOrRd } from 'd3-scale-chromatic';
import 'leaflet/dist/leaflet.css';

const BirdMap = ({ geoJson, predictions }) => {
  const colorScale = scaleSequential(interpolateYlOrRd).domain([0, 1]);

  const style = (feature) => {
    const regionName = feature.properties.landskap;
    const probability = predictions ? predictions[regionName] : 0;
    // Default to 0 if undefined
    const p = probability !== undefined ? probability : 0;
    
    return {
      fillColor: colorScale(p),
      weight: 1,
      opacity: 1,
      color: 'white',
      dashArray: '3',
      fillOpacity: 0.7
    };
  };

  const onEachFeature = (feature, layer) => {
    const regionName = feature.properties.landskap;
    const probability = predictions ? predictions[regionName] : undefined;
    const text = probability !== undefined ? `${(probability * 100).toFixed(1)}%` : 'No Data';
    layer.bindTooltip(`<strong>${regionName}</strong><br/>Probability: ${text}`);
  };

  return (
    <MapContainer center={[62.0, 15.0]} zoom={5} style={{ height: '100%', width: '100%' }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {geoJson && (
        <GeoJSON data={geoJson} style={style} onEachFeature={onEachFeature} key={JSON.stringify(predictions)} /> 
      )}
    </MapContainer>
  );
};
// Note: key={JSON.stringify(predictions)} forces re-render of GeoJSON when data changes, 
// which is sometimes necessary for style updates in react-leaflet.

export default BirdMap;
