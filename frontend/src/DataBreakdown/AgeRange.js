import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import "./Analysis.css"

const AgeRange = () => {
  const [ageRangeData, setAgeRangeData] = useState([]);
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');
  const [aiSummary, setAiSummary] = useState(""); // State for AI summary of data
  
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchAgeRange = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_age_range/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          if (data.age_range_frequency) {
            setAgeRangeData(data.age_range_frequency);
          } else {
            alert('No age range data found');
          }
        } catch (error) {
          alert('Error fetching age range data:', error);
        }
      };

      fetchAgeRange();
    }
  }, [selectedFile, selectedSheet]);
  
  // Fetch the AI response
  useEffect(() => {
    if (selectedFile && selectedSheet) {
        const aisummary = async () => {
            try {
            const response = await fetch(`http://localhost:5001/ai_summary/${selectedFile}/${selectedSheet}/Age Range`);
            const data = await response.json();
            if (data && data.aiSummary) {
                setAiSummary(data.aiSummary);
            } else {
                alert('No ai summary received');
            }
            } catch (error) {
            alert(`Error fetching summary: ${error}`);
            }
        };

        aisummary();
    }
}, [selectedFile, selectedSheet]); // Runs when selectedFile or selectedSheet changes

  // Convert the age range data into a format suitable for Recharts (array of objects)
  const chartData = ageRangeData ? Object.entries(ageRangeData).map(([age, count]) => ({
    age,
    count,
  })) : [];

  return (
    <div>
        <div className="content">
            <h1>Age Range Data for {selectedFile} - {selectedSheet}</h1>

            <div className="graph">
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

            <div>
                <h2>Summary</h2>
                <div>
                    {aiSummary ? (
                    <p>{aiSummary}</p>  // Display the summary if it's available
                    ) : (
                    <p>Loading summary...</p>  // Show loading message if summary is still being fetched
                    )}
                </div>
            </div>


        </div>

    </div>
  );
};

export default AgeRange;
