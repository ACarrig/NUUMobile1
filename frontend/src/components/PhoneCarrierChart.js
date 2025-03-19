import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

const CarrierChart = ({ top5Carriers, openWindow, selectedFile, selectedSheet }) => {
  return (
    <div className="summary-box">
      <h3>Top 5 Most Used Phone Carriers</h3>
      <div className="summary-graph">
        {top5Carriers && top5Carriers.carrier ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={Object.entries(top5Carriers.carrier)
              .map(([carrier, count]) => ({ carrier, count }))
              .sort((a, b) => b.count - a.count)}>
              <XAxis dataKey="carrier" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#C4D600" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p>Loading top 5 carriers...</p>
        )}
      </div>
      <button onClick={() => openWindow(`/sim_info?file=${selectedFile}&sheet=${selectedSheet}`)}>View More Sim Info</button>
    </div>
  );
};

export default CarrierChart;
