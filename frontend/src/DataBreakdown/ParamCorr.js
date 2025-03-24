import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'; 
import "./Analysis.css";
import '../components/Dashboard/Dashboard.css';

const ParamCorr = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');

    const [correlation, setCorrelation] = useState([]);
    const [aiSummary, setAiSummary] = useState("");

    useEffect(() => {
            if (selectedFile) {
                const fetchCorrelation = async () => {
                    try {
                        const response = await fetch(`http://localhost:5001/param_churn_correlation/${selectedFile}`);
                        const data = await response.json();
    
                        if (data.corr) {
                            setCorrelation(data.corr);
                        } else {
                            alert('No data found');
                        }
                    }
                    catch (error) {
                        alert('Error fetching defects:', error);
                    }
                }
                fetchCorrelation();
            }
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
            <div className="summary-box">
                <h3>Params Correlated with Churn</h3>
                {correlation && Object.keys(correlation).length ? (
                <div className="summary-graph">
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={Object.entries(correlation)
                            .map(([model, count]) => ({ model, count }))
                            .sort((a, b) => b.count - a.count)}>
                            <XAxis dataKey="model" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="count" fill="#C4D600" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
                ) : (
                    <p>Loading correlation data...</p>
                )}
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
