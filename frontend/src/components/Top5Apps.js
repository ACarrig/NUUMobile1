import React from 'react';
import './Dashboard.css';

const Top5Apps = ({ top5Apps, openWindow }) => {
  return (
    <div className="summary-box">
      <h3>Top 5 Most Used Apps</h3>
      {top5Apps && Object.keys(top5Apps).length ? (
        <ol>
          {Object.entries(top5Apps).sort((a, b) => b[1] - a[1]).map(([app, usage], index) => (
            <li key={index}>{app}: {usage} hrs</li>
          ))}
        </ol>
      ) : (
        <p>Loading top 5 apps...</p>
      )}
      <button onClick={() => openWindow('/appdata')}>View App Data</button>
    </div>
  );
};

export default Top5Apps;
