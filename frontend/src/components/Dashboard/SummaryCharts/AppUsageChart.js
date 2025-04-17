import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import '../Dashboard.css';

const AppUsageChart = ({ openWindow, selectedFile, selectedSheet }) => {
  const [appUsage, setAppUsage] = useState({}); // Local state for top 5 apps

  // Fetch top 5 apps when the component mounts
  useEffect(() => {
    const fetchAppUsage = async () => {
      try {
        const response = await fetch(`http://localhost:5001/get_app_usage/${selectedFile}/${selectedSheet}`);
        const data = await response.json();  
        if (data && data.app_frequency) {  
          setAppUsage(data.app_frequency);  
        }
      } catch (error) {
        console.error(`Error fetching top 5 apps: ${error}`);
      }
    };

    fetchAppUsage();
  }, [selectedFile, selectedSheet]); // Runs only once when the component mounts

  return (
    <div className="summary-box">
      <h3>Top 5 Most Used Apps (hrs) </h3>
      {appUsage && Object.keys(appUsage).length ? (
        <div className="summary-graph">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={Object.entries(appUsage)
              .map(([app, count]) => ({ app, count: (count / 3600).toFixed(2) })) 
              .sort((a, b) => b.count - a.count)
              .slice(0, 5)}>
              <XAxis dataKey="app" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#C4D600" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <p>Loading top 5 app...</p>
      )}
      <button onClick={() => openWindow(`/appdata?file=${selectedFile}&sheet=${selectedSheet}`)}>View App Data</button>
    </div>
  );
};

export default AppUsageChart;
