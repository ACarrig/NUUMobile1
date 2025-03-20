import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

const FeatureImportanceChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [featureImportances, setFeatureImportances] = useState([]);

    // Fetch Model Frequency from the selected file and sheet
    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchFeatureImportances = async () => {
                try {
                    console.log(`Fetching model frequency for file: ${selectedFile}, sheet: ${selectedSheet}`);
                    const response = await fetch(`http://localhost:5001/get_features`);
                    const data = await response.json();
                    if (data.features) {
                        setFeatureImportances(data.features); // Store model frequency data in state
                    }
                } catch (error) {
                    console.error(`Error fetching feature importances: ${error}`);
                }
            };

            fetchFeatureImportances();
        }
    }, [selectedFile, selectedSheet]);

    return (
        <div className="summary-box">
            <h3>Top 5 Features Importance</h3>
            {featureImportances && featureImportances.length ? (
                <div className="summary-graph">
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart
                            data={featureImportances
                                .sort((a, b) => b.Importance - a.Importance) // Sort by Importance
                                .slice(0, 5)} // Get top 5 features
                        >
                            <XAxis dataKey="Feature" />
                            <YAxis />
                            <Tooltip />
                            <Bar dataKey="Importance" fill="#C4D600" />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            ) : (
                <p>Loading top 5 feature importances...</p>
            )}
            <button onClick={() => openWindow(`/prediction?file=${selectedFile}&sheet=${selectedSheet}`)}>
                View Predictions
            </button>
        </div>
    );
};

export default FeatureImportanceChart;
