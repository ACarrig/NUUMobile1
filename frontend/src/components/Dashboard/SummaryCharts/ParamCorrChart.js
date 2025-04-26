import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import '../Dashboard.css';

const ParamCorrChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [correlation, setCorrelation] = useState([]);

    useEffect(() => {
        if (selectedFile) {
            const fetchCorrelation = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/param_churn_correlation/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();

                    if (data.corr) {
                        setCorrelation(data.corr);
                    } else {
                        console.log('No data found');
                    }
                }
                catch (error) {
                    console.log('Error fetching defects:', error);
                }
            }
            fetchCorrelation();
        }
    }, [selectedFile, selectedSheet]);

    return (
        <div className="summary-box">
            <h3>Params Correlated with Churn</h3>
            {correlation && Object.keys(correlation).length ? (
            <div className="summary-graph">
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={Object.entries(correlation)
                        .map(([corr, count]) => ({ corr, count }))
                        .sort((a, b) => b.count - a.count)
                        .slice(0, 5)}>
                        <XAxis dataKey="corr" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="count" fill="#C4D600" />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            ) : (
                <p>Loading correlation data...</p>
            )}
            <button onClick={() => openWindow(`/paramcorr?file=${selectedFile}&sheet=${selectedSheet}`)}>View Correlations</button>
        </div>
    );
};

export default ParamCorrChart;
