import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import '../Dashboard.css';

const AgeRangeChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [ageRange, setAgeRange] = useState({});

    // Fetch Age Range from backend
    useEffect(() => {
        if (selectedFile && selectedSheet) {
        const fetchAgeRange = async () => {
            try {
            console.log(`Fetching age range for file: ${selectedFile}, sheet: ${selectedSheet}`);
            const response = await fetch(`http://localhost:5001/get_age_range/${selectedFile}/${selectedSheet}`);
            const data = await response.json();
            if (data.age_range_frequency) {
                setAgeRange(data.age_range_frequency); // Store data in state
            }
            } catch (error) {
            console.error(`Error fetching age range: ${error}`);
            }
        };

        fetchAgeRange();
        }
    }, [selectedFile, selectedSheet]);

    // Function to get the highest age range
    const getHighestAgeRange = (ageRange) => {
        const maxCount = Math.max(...Object.values(ageRange));
        const maxAge = Object.keys(ageRange).find(age => ageRange[age] === maxCount);
        return `${maxAge}`;
    };

    // Function to get the lowest age range
    const getLowestAgeRange = (ageRange) => {
        const minCount = Math.min(...Object.values(ageRange));
        const minAge = Object.keys(ageRange).find(age => ageRange[age] === minCount);
        return `${minAge}`;
    };

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

          <div className="age-range-info">
            <p><strong>Highest Age Range: </strong>{getHighestAgeRange(ageRange)}</p>
            <p><strong>Lowest Age Range: </strong>{getLowestAgeRange(ageRange)}</p>
        </div>

        </div>
      ) : (
        <p>Loading age range data...</p>
      )}

      <button onClick={() => openWindow(`/agerange?file=${selectedFile}&sheet=${selectedSheet}`)}>View Age Range</button>
    </div>
  );
};

export default AgeRangeChart;
