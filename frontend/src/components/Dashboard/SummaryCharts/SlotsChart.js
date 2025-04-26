import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import '../Dashboard.css';

const SlotsChart = ({ openWindow, selectedFile, selectedSheet }) => {
  const [carrierData, setCarrierData] = useState([]);
  const [currentSlot, setCurrentSlot] = useState('Slot 1'); // State to track the selected slot

  // Fetch data for both Slot 1 and Slot 2
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchCarrierName = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_carrier_name_from_1slot/${selectedFile}/${selectedSheet}/${currentSlot}`);
          const data = await response.json();

          if (data.carrier) {
            // Update carrierData with the filtered data
            setCarrierData(data.carrier);
          }
        } catch (error) {
          console.log('Error fetching carrier name:', error);
        }
      };

      fetchCarrierName();
    }
  }, [selectedFile, selectedSheet, currentSlot]); // Runs when selectedFile, selectedSheet, or currentSlot changes

  // Toggle function to switch between Slot 1 and Slot 2
  const toggleSlot = () => {
    setCurrentSlot((prevSlot) => (prevSlot === 'Slot 1' ? 'Slot 2' : 'Slot 1'));
    setCarrierData([]);
  };

  return (
    <div className="summary-box">
      <div className="header-container">
        <h3>Top 5 Most Used Phone Carriers ({currentSlot})</h3> 
        <div>
          <button onClick={toggleSlot}>Change Slot</button>
        </div>
      </div>

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
      <button onClick={() => openWindow(`/slots_info?file=${selectedFile}&sheet=${selectedSheet}`)}>
        View More About the Sim Slots
      </button>
    </div>
  );
};

export default SlotsChart;
