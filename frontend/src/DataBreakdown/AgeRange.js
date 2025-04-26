import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import "./Analysis.css"
import AiSummary from './Summary';

const AgeRange = () => {
  const [ageRangeData, setAgeRangeData] = useState([]);
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');
  
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchAgeRange = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_age_range/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.age_range_frequency) {
            setAgeRangeData(data.age_range_frequency);
          } else {
            console.log('No age range data found');
          }
        } catch (error) {
          console.log('Error fetching age range data:', error);
        }
      };

      fetchAgeRange();
    }
  }, [selectedFile, selectedSheet]);

  // Convert the age range data into a format suitable for Recharts (array of objects)
  const chartData = ageRangeData ? Object.entries(ageRangeData).map(([age, count]) => ({
    age,
    count,
  })) : [];

  return (
    <div>
      <div className="content">
        <h1>Age Range Data for {selectedFile} - {selectedSheet}</h1>

        <div className="graph-container">
          {chartData.length > 0 && (
            <div>
              <h2>Age Range Frequency Chart</h2>
              <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData}>
                  <XAxis dataKey="age" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#C4D600" />
                  </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <AiSummary 
          selectedFile={selectedFile} 
          selectedSheet={selectedSheet} 
          selectedColumn={["Age Range"]}
        />

      </div>

    </div>
  );
};

export default AgeRange;
