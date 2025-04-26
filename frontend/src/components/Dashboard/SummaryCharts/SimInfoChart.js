import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import '../Dashboard.css';

const SimInfoChart = ({ openWindow, selectedFile, selectedSheet }) => {
  const [carrierData, setCarrierData] = useState([]);

  // Fetch top 5 most used carrier names
  useEffect(() => {
      if (selectedFile && selectedSheet) {
      const fetchCarrierName = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_carrier_name/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.carrier) {
            setCarrierData(data.carrier);
          }
        } catch (error) {
          console.log('Error fetching carrier name:', error);
        }
      };

      fetchCarrierName();
      }
    }, [selectedFile, selectedSheet]);

  return (
    <div className="summary-box">
      <h3>Top 5 Most Used Phone Carriers</h3>
      {carrierData && Object.keys(carrierData).length ? (
          <div className="summary-graph">
          <ResponsiveContainer width="100%" height={200}>
              <BarChart data={Object.entries(carrierData)
              .map(([carrier, count]) => ({ carrier, count }))
              .sort((a, b) => b.count - a.count)
              .slice(0, 5)}>
              <XAxis dataKey="carrier" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#C4D600" />
              </BarChart>
          </ResponsiveContainer>
          </div>
        ) : (
            <p>Loading top 5 phone carriers...</p>
        )}
      <button onClick={() => openWindow(`/sim_info?file=${selectedFile}&sheet=${selectedSheet}`)}>
        View More Sim Info
      </button>
    </div>
  );
};

export default SimInfoChart;
