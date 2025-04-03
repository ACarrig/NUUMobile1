import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import "./Predictions.css";

const ModelInfo = ({ selectedFile, selectedSheet }) => {
  const [featureImportances, setFeatureImportances] = useState([]);
  const [evalMetrics, setEvalMetrics] = useState(null);

  // Function to format unmapped feature names
  const formatFeatureName = (feature) => {
    return feature
      .replace(/_/g, " ") // Replace underscores with spaces
      .replace(/-/g, " ") // Replace dashes with spaces
      .replace(/\b\w/g, char => char.toUpperCase()); // Capitalize each word
  };

  // Fetch feature importances
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchFeatureImportances = async () => {
        try {
          console.log(`Fetching feature importances for file: ${selectedFile}, sheet: ${selectedSheet}`);
          const response = await fetch(`http://localhost:5001/get_features`);
          const data = await response.json();
          if (data.features) {
            const formattedFeatures = data.features
              .filter(feature => feature.Importance > 0)
              .map(feature => ({
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
  }, [selectedFile, selectedSheet]);

  // Fetch model evaluation metrics
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchEvalMetrics = async () => {
        try {
          const response = await fetch(`http://localhost:5001/get_eval/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          setEvalMetrics(data);
        } catch (error) {
          console.error('Error fetching evaluation metrics:', error);
        }
      };
      fetchEvalMetrics();
    }
  }, [selectedFile, selectedSheet]);

  return (
    <div className="model-info">
      <h2>Model Information</h2>

      {/* Feature Importance Graph */}
      {featureImportances.length > 0 && (
        <div className="model-container">
          <h3>Feature Importances</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={featureImportances} layout="vertical">
              <XAxis type="number" />
              <YAxis dataKey="Feature" type="category" width={200} interval={0} />
              <Tooltip />
              <Bar dataKey="Importance" fill="#C4D600" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model Evaluation Metrics */}
      <div className="model-container">
        <h3>Model Evaluation</h3>

        {/* Model Evaluation Matrix */}
        {evalMetrics ? (
          <div className='model-eval-container'>
            <div className='eval-box'>
              <strong>Accuracy:</strong> {evalMetrics.accuracy != null ? (evalMetrics.accuracy * 100).toFixed(2) : "N/A"}%
            </div>
            <div className='eval-box'>
              <strong>Precision:</strong> {evalMetrics.precision != null ? (evalMetrics.precision * 100).toFixed(2) : "N/A"}%
            </div>
            <div className='eval-box'>
              <strong>Recall:</strong> {evalMetrics.recall != null ? (evalMetrics.recall * 100).toFixed(2) : "N/A"}%
            </div>
            <div className='eval-box'>
              <strong>F1 Score:</strong> {evalMetrics.f1_score != null ? evalMetrics.f1_score.toFixed(4) : "N/A"}
            </div>
          </div>
        ) : (
          <p>Loading evaluation metrics...</p>
        )}

        {/* Confusion Matrix */}
        {evalMetrics && evalMetrics.confusion_matrix_image ? (
          <div className='confusion-matrix'>
              <img
                src={`data:image/png;base64,${evalMetrics.confusion_matrix_image}`}
                alt="Confusion Matrix"
                style={{ width: '100%', maxWidth: '500px', height: 'auto' }}
              />
          </div>
        ) : (
          <p>Loading confusion matrix...</p>
        )}

      </div>

    </div>
  );
};

export default ModelInfo;
