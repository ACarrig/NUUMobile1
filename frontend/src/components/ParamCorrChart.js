import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import './Dashboard.css';

const ParamCorrChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [correlation, setCorrelation] = useState([]);

    useEffect(() => {
        const fetchCorrelation = async () => {
            try {
                const response = await fetch(`http://localhost:5001/param_churn_correlation/${selectedFile}`);
                const data = await response.json();

                if (data.corr) {
                    const sortedData = Object.entries(data.corr)
                        .map(([param, value]) => ({ param, value }))
                        .sort((a, b) => b.value - a.value)
                        .slice(0, 3);
                    setCorrelation(sortedData);
                } else {
                    alert('No data found');
                }
            }
            catch (error) {
                alert('Error fetching defects:', error);
            }

            fetchCorrelation();
        }
    }, [selectedFile, selectedSheet]);

    return (
        <div className="summary-box">
            <h3>Params Correlated with Churn</h3>
            {correlation.length ? (
                <div className="summary-graph">
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart 
                            data={correlation} 
                            layout="vertical"
                            margin={{ left: 80, right: 30, top: 10, bottom: 10 }}>
                            <XAxis type="number" />
                            <YAxis dataKey="param" type="category" width={150} />
                            <Tooltip />
                            <Bar dataKey="value" fill="#C4D600" />
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
