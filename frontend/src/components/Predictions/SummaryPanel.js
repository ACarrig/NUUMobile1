import React from 'react';
import "./Predictions.css";

const SummaryPanel = ({ filteredPredictionData }) => {
  if (!filteredPredictionData || filteredPredictionData.length === 0) {
    return (
      <div className="summary-panel">
        <p>Loading predictions...</p>
      </div>
    );
  }

  const churnCount = filteredPredictionData.filter(p => p["Churn Prediction"] === 1).length;
  const notChurnCount = filteredPredictionData.filter(p => p["Churn Prediction"] === 0).length;
  const churnRate = ((churnCount / filteredPredictionData.length) * 100).toFixed(2);

  const avgProbability = (filteredPredictionData.reduce((acc, p) => acc + p["Churn Probability"], 0) / filteredPredictionData.length * 100).toFixed(2);
  const maxProbability = (Math.max(...filteredPredictionData.map(p => p["Churn Probability"])) * 100).toFixed(2);
  const minProbability = (Math.min(...filteredPredictionData.map(p => p["Churn Probability"])) * 100).toFixed(2);

  const probabilityDistribution = {
    "0-20%": filteredPredictionData.filter(p => p["Churn Probability"] <= 0.2).length,
    "20-40%": filteredPredictionData.filter(p => p["Churn Probability"] > 0.2 && p["Churn Probability"] <= 0.4).length,
    "40-60%": filteredPredictionData.filter(p => p["Churn Probability"] > 0.4 && p["Churn Probability"] <= 0.6).length,
    "60-80%": filteredPredictionData.filter(p => p["Churn Probability"] > 0.6 && p["Churn Probability"] <= 0.8).length,
    "80-100%": filteredPredictionData.filter(p => p["Churn Probability"] > 0.8).length,
  };

  return (
    <div className="summary-panel">
      <h2>Summary</h2>
      <p><strong>Total Rows:</strong> {filteredPredictionData.length}</p>

      <div className="summary-item">
        <p><strong>Churn Predictions:</strong></p>
        <ul>
          <li><strong>Churn (1):</strong> {churnCount}</li>
          <li><strong>Not Churn (0):</strong> {notChurnCount}</li>
          <li><strong>Churn Rate:</strong> {churnRate}%</li>
        </ul>
      </div>

      <div className="summary-item">
        <p><strong>Churn Probability Stats:</strong></p>
        <ul>
          <li><strong>Average Probability:</strong> {avgProbability}%</li>
          <li><strong>Max Probability:</strong> {maxProbability}%</li>
          <li><strong>Min Probability:</strong> {minProbability}%</li>
        </ul>
      </div>

      <div className="summary-item">
        <p><strong>Churn Probability Distribution:</strong></p>
        <ul>
          {Object.entries(probabilityDistribution).map(([range, count]) => (
            <li key={range}>{range}: {count}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default SummaryPanel;
