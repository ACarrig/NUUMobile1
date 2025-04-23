import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import '../Dashboard.css';

const ModelTypeChart = ({ openWindow, selectedFile, selectedSheet }) => {
  const [modelType, setModelType] = useState([]); // State for model frequency data

  // Fetch Model Frequency from the selected file and sheet
  useEffect(() => {
      if (selectedFile && selectedSheet) {
      const fetchModelType = async () => {
          try {
          console.log(`Fetching model frequency for file: ${selectedFile}, sheet: ${selectedSheet}`);
          const response = await fetch(`http://localhost:5001/get_model_type/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.model) {
              setModelType(data.model); // Store model frequency data in state
          }
          } catch (error) {
          console.error(`Error fetching age range: ${error}`);
          }
      };

      fetchModelType();
      }
  }, [selectedFile, selectedSheet]);

  return (
    <div className="summary-box">
      <h3>Top 5 Most Used Models</h3>
      {modelType && Object.keys(modelType).length ? (
        <div className="summary-graph">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={Object.entries(modelType)
              .map(([model, count]) => ({ model, count }))
              .sort((a, b) => b.count - a.count)
              .slice(0, 5)}>
              <XAxis dataKey="model" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#C4D600" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p>Loading top 5 model types...</p>
      )}
      <button onClick={() => openWindow(`/modeltype?file=${selectedFile}&sheet=${selectedSheet}`)}>View More Model Frequency</button>
    </div>
  );
};

export default ModelTypeChart;
