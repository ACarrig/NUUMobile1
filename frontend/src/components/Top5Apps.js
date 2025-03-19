import React, { useEffect, useState } from 'react';
import './Dashboard.css';

const Top5Apps = ({ openWindow }) => {
  const [top5Apps, setTop5Apps] = useState({}); // Local state for top 5 apps

  // Fetch top 5 apps when the component mounts
  useEffect(() => {
    const fetchTop5Apps = async () => {
      try {
        const response = await fetch('http://localhost:5001/top5apps');
        const data = await response.json();  
        if (data && data.top_5_apps) {  
          setTop5Apps(data.top_5_apps);  
        }
      } catch (error) {
        console.error(`Error fetching top 5 apps: ${error}`);
      }
    };

    fetchTop5Apps();
  }, []); // Runs only once when the component mounts

  return (
    <div className="summary-box">
      <h3>Top 5 Most Used Apps</h3>
      {top5Apps && Object.keys(top5Apps).length ? (
        <ol>
          {Object.entries(top5Apps)
            .sort((a, b) => b[1] - a[1])
            .map(([app, usage], index) => (
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
