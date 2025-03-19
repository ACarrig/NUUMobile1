import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

const AgeRangeChart = ({ ageRange, getHighestAgeRange, getLowestAgeRange, openWindow, selectedFile, selectedSheet }) => {
  return (
    <div className="summary-box">
      <h3>Age Range Frequency</h3>
      {ageRange && Object.keys(ageRange).length > 0 ? (
        <div className="summary-graph">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={Object.entries(ageRange).map(([age, count]) => ({ age, count }))}>
              <XAxis dataKey="age" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#C4D600" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p>Loading age range data...</p>
      )}
      <div className="age-range-info">
        <p><strong>Highest Age Range: </strong>{getHighestAgeRange(ageRange)}</p>
        <p><strong>Lowest Age Range: </strong>{getLowestAgeRange(ageRange)}</p>
      </div>
      <button onClick={() => openWindow(`/agerange?file=${selectedFile}&sheet=${selectedSheet}`)}>View Age Range</button>
    </div>
  );
};

export default AgeRangeChart;
