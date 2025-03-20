import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'; 
import "./Analysis.css";

const ParamCorr = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');

    const [correlation, setCorrelation] = useState([]);
    const [aiSummary, setAiSummary] = useState("");

    useEffect(() => {
        const fetchCorrData = async () => {
            if (!selectedFile) return;

            try {
                const response = await fetch(`http://localhost:5001/param_churn_correlation/${selectedFile}`);
                const data = await response.json();

                if (data.corr) {
                    const sortedData = Object.entries(data.corr)
                        .map(([param, value]) => ({ param, value }))
                        .sort((a, b) => b.value - a.value);
                    setCorrelation(sortedData);
                } else {
                    console.warn('No data found for correlation');
                }
            } catch (error) {
                console.error('Error fetching correlation data:', error);
            }
        };

        fetchCorrData();
    }, [selectedFile]);

    useEffect(() => {
        const fetchAiSummary = async () => {
            if (!selectedFile) return;

            try {
                const response = await fetch(`http://localhost:5001/churn_corr_summary/${selectedFile}`);
                const data = await response.json();

                if (data && data.aiSummary) {
                    setAiSummary(data.aiSummary);
                } else {
                    console.warn('No AI summary received');
                }
            } catch (error) {
                console.error('Error fetching AI summary:', error);
            }
        };

        fetchAiSummary();
    }, [selectedFile]);

    return (
        <div className="content">
            <div className="graph-container">
                <div className="chart">
                    {correlation.length > 0 ? (  
                        <div className="summary-graph">
                            <ResponsiveContainer width="100%" height={500}>
                                <BarChart 
                                    data={correlation} 
                                    layout="vertical" 
                                    margin={{ left: 80, right: 30, top: 10, bottom: 10 }}
                                >
                                    <XAxis type="number" />
                                    <YAxis dataKey="param" type="category" width={150} />
                                    <Tooltip />
                                    <Bar dataKey="value" fill="#C4D600" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    ) : (
                        <p>Loading Parameter Correlation to Churn...</p>
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

export default ParamCorr;
