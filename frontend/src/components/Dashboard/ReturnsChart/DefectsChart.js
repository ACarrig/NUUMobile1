import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import '../Dashboard.css';

const DefectsChart = ({ selectedFile, selectedSheet }) => {
  const [topDefects, setTopDefects] = useState([]);

  useEffect(() => {
    if (selectedFile && selectedSheet) {
    const fetchDefects = async () => {
      try {
      const response = await fetch(`http://localhost:5001/device_return_info/${selectedFile}/${selectedSheet}`);
      const data = await response.json();
      if (data.defects) {
        setTopDefects(data.defects);
      } else {
        console.log('No defects data found');
      }
      } catch (error) {
        console.log('Error fetching defects:', error);
      }
    };

    fetchDefects();
    }
  }, [selectedFile, selectedSheet]);

  return (
    <div className="summary-box">
      <h3>Returns Defects</h3>
      <p>Verified issues from the testing team</p>

      {topDefects && Object.keys(topDefects).length ? (
      <div className="summary-graph">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={Object.entries(topDefects)
            .map(([defects, count]) => ({ defects, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5)}>
            <XAxis dataKey="defects" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#C4D600" />
          </BarChart>
        </ResponsiveContainer>
      </div>
        ) : (
          <p>Loading Defects...</p>
        )}
        </div>
  );
};

export default DefectsChart;
