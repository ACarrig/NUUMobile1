import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import "./Predictions.css";

const modelToEndpoint = {
  ensemble: 'em',
  xgb: 'xgb',
  nn: 'nn',
};

const ModelInfo = ({ selectedFile, selectedSheet, selectedModel }) => {
  const [featureImportances, setFeatureImportances] = useState([]);
  const [evalMetrics, setEvalMetrics] = useState(null);
  const [hideZeroImportance, setHideZeroImportance] = useState(false);

  const formatFeatureName = (feature) => {
    return feature
      .replace(/_/g, " ") // underscores → spaces
      .replace(/-/g, " ") // dashes → spaces
      .replace(/\b\w/g, char => char.toUpperCase()); // capitalize words
  };

  useEffect(() => {
    if (!selectedFile || !selectedSheet || !selectedModel) {
      setFeatureImportances([]);
      return;
    }
  
    const fetchFeatureImportances = async () => {
      setFeatureImportances([]); // clear old data
  
      try {
        const endpointPrefix = modelToEndpoint[selectedModel] || 'em';
        const response = await fetch(
          `http://localhost:5001/${endpointPrefix}_get_features/${selectedFile}/${selectedSheet}`
        );
        const data = await response.json();
  
        if (data.features) {
          const formattedFeatures = data.features.map(f => ({
            Feature: formatFeatureName(f.Feature),
            Importance: parseFloat(f.Importance.toFixed(4)),
          }));
          setFeatureImportances(formattedFeatures);
        } else {
          setFeatureImportances([]);
        }
      } catch (error) {
        console.error('Error fetching feature importances:', error);
        setFeatureImportances([]);
      }
    };
  
    fetchFeatureImportances();
  }, [selectedFile, selectedSheet, selectedModel]);  

  useEffect(() => {
    if (!selectedFile || !selectedSheet || !selectedModel) {
      setEvalMetrics(null);
      return;
    }
  
    const fetchEvalMetrics = async () => {
      setEvalMetrics(null); // Clear old data before fetching
  
      try {
        const endpointPrefix = modelToEndpoint[selectedModel] || 'em';
        const response = await fetch(
          `http://localhost:5001/${endpointPrefix}_get_eval/${selectedFile}/${selectedSheet}`
        );
        const data = await response.json();
        setEvalMetrics(data);
      } catch (error) {
        console.error('Error fetching evaluation metrics:', error);
        setEvalMetrics(null);
      }
    };
  
    fetchEvalMetrics();
  }, [selectedFile, selectedSheet, selectedModel]);  

  const visibleFeatures = hideZeroImportance
    ? featureImportances.filter(f => f.Importance !== 0)
    : featureImportances;

  const chartHeight = Math.max(200, visibleFeatures.length * 30);

  return (
    <div className="model-info">
      <h2>Infomation about the Model</h2>

      {/* Feature Importance Graph */}
        <div className="model-container">
          <div className="feature-header-container">
            <div className="icon-container">
              <span className="feature-icon iconify" data-icon="solar:chart-bold" data-inline="false"></span>
              <h3>Feature Importances</h3>
            </div>
            <div className="checkbox-container">
              <label>
                <input
                  type="checkbox"
                  checked={hideZeroImportance}
                  onChange={() => setHideZeroImportance(!hideZeroImportance)}
                />
                Hide zero importance
              </label>
            </div>
          </div>
          <p>This displays the top features driving churn predictions, based on the selected model's learned importances.</p>
          <br/>
          {featureImportances.length > 0 ? (
            <ResponsiveContainer width="100%" height={chartHeight}>
              <BarChart data={visibleFeatures} layout="vertical">
                <XAxis type="number" />
                <YAxis dataKey="Feature" type="category" width={180} interval={0} />
                <Tooltip />
                <Bar dataKey="Importance" fill="#C4D600" />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p>Loading Feature Importances...</p>
          )}
        </div>

      {/* Model Evaluation Metrics */}
      <div className="model-container">

        <div className="icon-container">
            <span className="eval-icon iconify" data-icon="prime:gauge" data-inline="false"></span>
            <h3>Model Evaluation</h3>
          </div>

        {/* Model Evaluation Matrix */}
        <div className='model-eval-container'>
          {evalMetrics?.error ? (
            <div className="error-message">
              <h4>Model Evaluation Unavailable</h4>
              <p>{evalMetrics.error}</p>
            </div>
          ) : evalMetrics ? (
            <div className='model-eval-metric-container'>
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
          {evalMetrics?.error ? (
            <></>
          ) : evalMetrics && evalMetrics.confusion_matrix_image ? (
            <div className='confusion-matrix'>
                <img
                  src={`data:image/png;base64,${evalMetrics.confusion_matrix_image}`}
                  alt="Confusion Matrix"
                  style={{ width: '100%', height: 'auto' }}
                />
            </div>
          ) : (
            <p>Loading confusion matrix...</p>
          )}

        </div>

      </div>

    </div>
  );
};

export default ModelInfo;
