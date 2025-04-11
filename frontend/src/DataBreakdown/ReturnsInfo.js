import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
// import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts'; 
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import "./Analysis.css";
import AiSummary from "./Summary"

const ReturnsInfo = () => {
    const [returnsData, setReturnsData] = useState([]);
    const [numReturns, setNumReturns] = useState(0);

    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchReturnsData = async () => {
              try {
                const response = await fetch(`http://localhost:5001/device_return_info/${selectedFile}/${selectedSheet}`);
                const data = await response.json();
                if (data.defects) {
                  setReturnsData(data.defects);
                } else {
                  alert('No defects data found');
                }
              } catch (error) {
                alert('Error fetching returns data:', error);
              }
            };

            fetchReturnsData();
        }
    }, [selectedFile, selectedSheet]);


      useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchNumReturns = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/num_returns/${selectedFile}/${selectedSheet}`)
                    const data = await response.json();
                    if (data && data.num_returns) {
                        setNumReturns(data.num_returns);
                    } else {
                        alert('No defect count received');
                    }
                } catch (error) {
                    alert('Error fetching returns data:', error);
                }
            };

            fetchNumReturns();
        }
    }, [selectedFile, selectedSheet]);

    return (
        <div className="content">
            <h1>Return Info for {selectedFile} - {selectedSheet}</h1>

            <div className="summary-container">
                {numReturns ? (
                <p><strong>Total Returns in this Data Set:</strong> {numReturns}</p>
                ) : (
                <p>Loading total...</p>
                )}
            </div>

            {/* Bar Chart for Defects Types */}
            <div className="graph-container">
                <div className="chart">
                    <h2>Reasons for Return</h2>
                    <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={Object.entries(returnsData)
                        .map(([defects, count]) => ({ defects, count }))
                        .sort((a, b) => b.count - a.count)} >
                        <XAxis dataKey="defects" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#C4D600" />
                    </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            <AiSummary 
                selectedFile={selectedFile} 
                selectedSheet={selectedSheet} 
                selectedColumn={["Defect / Damage type"]}
            />

        </div>
    );
};

export default ReturnsInfo;
