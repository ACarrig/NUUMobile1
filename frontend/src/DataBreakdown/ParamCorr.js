import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'; 
import "./Analysis.css";

const ParamCorr = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    const [correlation, setCorrelation] = useState([]);
    const [aiSummary, setAiSummary] = useState("");

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchCorrData = async () => {
                try {
                const response = await fetch(`http://localhost:5001/param_churn_correlation/${selectedFile}`);
                const data = await response.json();
                if (data.corr) {
                    const sortedData = Object.entries(data.corr)
                        .map(([param, value]) => ({ param, value }))
                        .sort((a, b) => b.value - a.value);
                    setCorrelation(sortedData);
                } else {
                    alert('No data found');
                }
                } catch (error) {
                alert('Error fetching correlation data:', error);
                }
            };

            fetchCorrData();
        }
    }, [selectedFile, selectedSheet]);

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const aisummary = async () => {
            try {
                const response = await fetch(`http://localhost:5001/churn_corr_summary/${selectedFile}`);
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
            <div className="graph-container">
                <div className="chart">
                    {correlation && Object.keys(correlation).length ? (
                    <div className="summary-graph">
                        <ResponsiveContainer width="100%" height={500}>
                            <BarChart 
                                data={correlation} 
                                layout="vertical">
                                <XAxis type="number" />
                                <YAxis dataKey="param" type="category" width={250} />
                                <Tooltip />
                                <Bar dataKey="value" fill="#C4D600" />
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

export default ParamCorr;
