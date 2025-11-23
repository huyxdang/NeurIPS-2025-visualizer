/* frontend/src/Map.js */
import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import './App.css'; // We will define pirate styles here

const App = () => {
  const [data, setData] = useState([]);
  const [selectedPaper, setSelectedPaper] = useState(null);

  // 1. Load Data
  useEffect(() => {
    fetch('/neurips_data.json')
      .then(response => response.json())
      .then(jsonData => setData(jsonData));
  }, []);

  // 2. Define the Plotly Data Trace
  // We use 'scattergl' for WebGL rendering (handles 5k+ points smoothly)
  const plotData = [
    {
      x: data.map(d => d.umap_x),
      y: data.map(d => d.umap_y),
      text: data.map(d => d.paper), // Shows on hover
      mode: 'markers',
      type: 'scattergl',
      marker: {
        size: 8,
        color: data.map(d => d.cluster || 0), // Color by cluster (use cluster column, default to 0)
        colorscale: 'Earth', // Earthy tones for pirate vibe
        opacity: 0.8,
        line: { color: '#3e2723', width: 1 } // Dark border around dots
      },
      hovertemplate: 
        '<b>%{text}</b><br>' +
        '<i>Click point to read scroll...</i><extra></extra>',
    }
  ];

  // 3. Define the Layout (The Map Look)
  const layout = {
    title: '',
    showlegend: false,
    autosize: true,
    hovermode: 'closest',
    dragmode: 'pan', // Default interaction is panning, not selecting
    paper_bgcolor: '#f4e4bc', // Parchment color matching background
    plot_bgcolor: 'rgba(0,0,0,0)', // Transparent to show image
    margin: { l: 0, r: 0, b: 0, t: 0 }, // Full screen
    
    // Hide Axes (It's a map, not a chart)
    xaxis: { visible: false, showgrid: false, range: [-5, 15] }, 
    yaxis: { visible: false, showgrid: false, range: [-5, 15] },

    // THE PIRATE MAP BACKGROUND
    images: [
      {
        source: "/map-background.png", // Background image from public folder
        xref: "x",
        yref: "y",
        x: -10,  // Adjust these based on your UMAP min/max
        y: 20,   // Adjust these based on your UMAP min/max
        sizex: 30,
        sizey: 30,
        sizing: "stretch",
        opacity: 1,
        layer: "below"
      }
    ]
  };

  return (
    <div className="pirate-container">
      
      {/* THE MAP AREA */}
      <div className="map-wrapper">
        <Plot
          data={plotData}
          layout={layout}
          useResizeHandler={true}
          style={{ width: "100%", height: "100%" }}
          onClick={(eventData) => {
            // Get the index of the clicked point
            const pointIndex = eventData.points[0].pointNumber;
            setSelectedPaper(data[pointIndex]);
          }}
        />
      </div>

      {/* THE CAPTAIN'S LOG (Sidebar) */}
      <div className={`captains-log ${selectedPaper ? 'open' : ''}`}>
        {selectedPaper ? (
          <>
            <button className="close-btn" onClick={() => setSelectedPaper(null)}>X</button>
            <h2>📜 {selectedPaper.paper}</h2>
            <div className="cluster-tag">Island: {selectedPaper.cluster_name}</div>
            
            <h3>The Problem</h3>
            <p>{selectedPaper.problem}</p>
            
            <h3>The Solution</h3>
            <p>{selectedPaper.solution}</p>
            
            <h3>ELI5 (For the Crew)</h3>
            <p><i>{selectedPaper.eli5}</i></p>
            
            <a href={selectedPaper.link} target="_blank" rel="noreferrer" className="read-btn">
              Read Full Scroll ↗
            </a>
          </>
        ) : (
          <div className="placeholder-text">
            <h3>🗺️ NeurIPS Archipelago</h3>
            <p>Click on a location to inspect the research hidden there.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;