import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import './Dashboard.css';

// Function to format unmapped feature names
const formatFeatureName = (feature) => {
    return feature
        .replace(/_/g, " ") // Replace underscores with spaces
        .replace(/-/g, " ") // Replace dashes with spaces
        .replace(/\b\w/g, char => char.toUpperCase()); // Capitalize each word
};

const FeatureImportanceChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [featureImportances, setFeatureImportances] = useState([]);
    const [selectedModel, setSelectedModel] = useState('ensemble'); // Default to ensemble

    // Model options
    const modelOptions = [
        { value: 'ensemble', label: 'Ensemble Model' },
        { value: 'nn', label: 'Neural Network' }
    ];

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchFeatureImportances = async () => {
                try {
                    const endpointPrefix = selectedModel === 'ensemble' ? 'em' : 'nn';
                    const response = await fetch(
                        `http://localhost:5001/${endpointPrefix}_get_features/${selectedFile}/${selectedSheet}`
                    );
                    const data = await response.json();
                    if (data.features) {
                        // Apply user-friendly formatting
                        const formattedFeatures = data.features.map(feature => ({
                            Feature: formatFeatureName(feature.Feature),
                            Importance: parseFloat(feature.Importance.toFixed(4))
                        }));
                        setFeatureImportances(formattedFeatures);
                    }
                } catch (error) {
                    console.error(`Error fetching feature importances: ${error}`);
                }
            };

            fetchFeatureImportances();
        }
    }, [selectedFile, selectedSheet, selectedModel]); // Added selectedModel to dependencies

    const handleModelSelectChange = (event) => {
        setSelectedModel(event.target.value);
    };

    return (
        <div className="summary-box">
            <div className="feature-header">
                <h3>Top 5 Feature Importances</h3>
                {selectedFile && selectedSheet && (
                    <div className="model-selector">
                        <label htmlFor="model-select">Model:</label>
                        <select 
                            id="model-select"
                            value={selectedModel}
                            onChange={handleModelSelectChange}
                        >
                            {modelOptions.map(option => (
                                <option key={option.value} value={option.value}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </div>
                )}
            </div>
            
            {featureImportances && featureImportances.length ? (
                <div className="summary-graph">
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={featureImportances
                            .sort((a, b) => b.Importance - a.Importance)
                            .slice(0, 5)}>
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