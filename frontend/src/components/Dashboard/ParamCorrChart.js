import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import './Dashboard.css';

const ParamCorrChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [correlation, setCorrelation] = useState([]);

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

    return (
        <div className="summary-box">
            <h3>Params Correlated with Churn</h3>
            {correlation && Object.keys(correlation).length ? (
            <div className="summary-graph">
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={Object.entries(correlation)
                        .map(([model, count]) => ({ model, count }))
                        .sort((a, b) => b.count - a.count)
                        .slice(1, 6)}>
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
            <button onClick={() => openWindow('/paramcorr')}>View Correlations</button>
        </div>
    );
};

export default ParamCorrChart;
