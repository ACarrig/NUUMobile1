// Page to have a component for top defects / returns info, when previewing on dashboard

import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { 
    BarChart, Bar, XAxis, YAxis, ResponsiveContainer, 
    PieChart, Pie, Cell, Tooltip, Legend 
} from 'recharts';
import './Dashboard.css';

const DefectsChart = ({ openWindow, selectedFile, selectedSheet }) => {
  const [topDefects, setTopDefects] = useState([]);

  useEffect(() => {
    if (selectedFile && selectedSheet) {
    const fetchDefects = async () => {
      try {
      const response = await fetch(`http://localhost:5001/device_return_info/${selectedFile}/${selectedSheet}`);
      const data = await response.json();
      if (data.defects) {
        const formattedData = Object.entries(data.defects)
          .map(([name, count]) => ({ name, count }))
          .sort((a, b) => b.count - a.count)
          .slice(0, 5);
        setTopDefects(formattedData);
      } else {
        alert('No defects data found');
      }
      } catch (error) {
      alert('Error fetching carrier name:', error);
      }
    };

    fetchDefects();
    }
  }, [selectedFile, selectedSheet]);

  return (
    <div className="summary-box">
      <h3>Returns Defects</h3>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={topDefects}>
          <XAxis dataKey="defects" tick={{ fontSize: 12 }} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="number" fill="#C4D600" />
        </BarChart>
      </ResponsiveContainer>
      <button onClick={() => openWindow('/returnsinfo')}>View Return Data</button>
    </div>
  );
};

export default DefectsChart;
