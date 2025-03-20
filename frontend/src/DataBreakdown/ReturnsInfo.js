import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts'; 
import "./Analysis.css";

const ReturnsInfo = () => {
    const [aiSummary, setAiSummary] = useState("");
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

    useEffect(() => {
        if (selectedFile && selectedSheet) {
          const aisummary = async () => {
            try {
                const response = await fetch(`http://localhost:5001/device_returns_summary/${selectedFile}/${selectedSheet}`);
                const data = await response.json();
                if (data && data.aiSummary) {
                setAiSummary(data.aiSummary);
                } else {
                alert('No AI summary received');
                }
            } catch (error) {
                alert('error getting summary');
            }
          };
    
          aisummary();
        }
    }, [selectedFile, selectedSheet]);


    return (
        <div className="content">
            <div className="summary-container">
                {numReturns ? (
                <h2>Total Returns in this Data Set: {numReturns}</h2>
                ) : (
                <p>Loading total...</p>
                )}
            </div>

            <div className="graph-container">
                <div className="chart">
                    {returnsData && Object.keys(returnsData).length ? (
                    <div className="summary-graph">
                        <ResponsiveContainer width="100%" height={500}>
                            <BarChart data={Object.entries(returnsData)
                            .map(([model, count]) => ({ model, count }))
                            .sort((a, b) => b.count - a.count)}>
                            <XAxis dataKey="model" tick={{ angle: 0, textAnchor: 'end' }}/>
                            <YAxis />
                            <BarTooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    ) : (
                    <p>Loading Defects...</p>
                    )}
                </div>
            </div>

            <div className="summary-container">
                <h2>Summary</h2>
                <div>
                    {aiSummary ? (
                    <p>{aiSummary}</p>
                    ) : (
                    <p>Loading summary...</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ReturnsInfo;
