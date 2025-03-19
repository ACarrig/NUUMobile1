import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import './Dashboard.css';

const ParamCorrChart = ({ openWindow, selectedFile }) => {
    const [correlation, setCorrelation] = useState([]);

    useEffect(() => {
        const fetchCorrelation = async () => {
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
            }
            catch (error) {
                alert('Error fetching defects:', error);
            }

            fetchCorrelation();
        }
    }, [selectedFile]);

    return (
        <div className="summary-graph">
            <ResponsiveContainer width="100%" height={400}>
                <BarChart 
                    data={correlation} 
                    layout="vertical"
                >
                    <XAxis type="number" />
                    <YAxis dataKey="param" type="category" width={150} />
                    <Tooltip />
                    <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
};

export default ParamCorrChart;
