import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

const ModelFrequencyChart = ({ modelFrequency, openWindow, selectedFile, selectedSheet }) => {
  return (
    <div className="summary-box">
      <h3>Top 5 Most Used Models</h3>
      {modelFrequency ? (
        <div className="summary-graph">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={Object.entries(modelFrequency)
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

export default ModelFrequencyChart;
