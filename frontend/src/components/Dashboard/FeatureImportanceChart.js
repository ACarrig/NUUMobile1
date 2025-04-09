import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

// Mapping technical feature names to user-friendly names
const FEATURE_NAME_MAPPING = {
    "interval - activate": "Time Since Activation",
    "last_boot - activate": "Time Since Last Boot",
    "interval - last_boot": "Interval Between Boots",
    "Warranty_Yes": "Under Warranty",
    "sim_info_status_uninserted": "SIM Not Inserted"
};

// Function to format unmapped feature names
const formatFeatureName = (feature) => {
    if (FEATURE_NAME_MAPPING[feature]) {
        return FEATURE_NAME_MAPPING[feature]; // Use predefined mapping if available
    }
    return feature
        .replace(/_/g, " ") // Replace underscores with spaces
        .replace(/-/g, " ") // Replace dashes with spaces
        .replace(/\b\w/g, char => char.toUpperCase()); // Capitalize each word
};

const FeatureImportanceChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [featureImportances, setFeatureImportances] = useState([]);

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchFeatureImportances = async () => {
                try {
                    console.log(`Fetching model frequency for file: ${selectedFile}, sheet: ${selectedSheet}`);
                    const response = await fetch(`http://localhost:5001/get_features/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();
                    if (data.features) {
                        // Apply user-friendly formatting
                        const formattedFeatures = data.features.map(feature => ({
                            Feature: formatFeatureName(feature.Feature), // Apply formatting function
                            Importance: parseFloat(feature.Importance.toFixed(4)) // Round to 4 decimal places
                        }));
                        setFeatureImportances(formattedFeatures);
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
            <h3>Top 5 Feature Importances</h3>
            {featureImportances && featureImportances.length ? (
                <div className="summary-graph">
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart
                            data={featureImportances
                                .sort((a, b) => b.Importance - a.Importance)
                                .slice(0, 5)}
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
            <button onClick={() => openWindow(`/predictions?file=${selectedFile}&sheet=${selectedSheet}`)}>
                View Predictions
            </button>
        </div>
    );
};

export default FeatureImportanceChart;
