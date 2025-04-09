import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import "./Predictions.css";

const SummaryPanel = ({ filteredPredictionData }) => {
  if (!filteredPredictionData || filteredPredictionData.length === 0) {
    return (
      <div className="summary-panel">
        <p>Loading summary of predictions...</p>
      </div>
    );
  }

  const churnCount = filteredPredictionData.filter((p) => p["Churn Prediction"] === 1).length;
  const notChurnCount = filteredPredictionData.filter((p) => p["Churn Prediction"] === 0).length;
  const churnRate = ((churnCount / filteredPredictionData.length) * 100).toFixed(2);

  const avgProbability = (
    (filteredPredictionData.reduce((acc, p) => acc + p["Churn Probability"], 0) / filteredPredictionData.length) *
    100
  ).toFixed(2);
  const maxProbability = (Math.max(...filteredPredictionData.map((p) => p["Churn Probability"])) * 100).toFixed(2);
  const minProbability = (Math.min(...filteredPredictionData.map((p) => p["Churn Probability"])) * 100).toFixed(2);

  // Define bins dynamically for histogram
  const bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]; // 5 bins
  const histogramData = bins.slice(0, -1).map((bin, index) => {
    const count = filteredPredictionData.filter(
      (p) => p["Churn Probability"] >= bin && p["Churn Probability"] < bins[index + 1]
    ).length;
    return { range: `${(bin * 100).toFixed(0)}-${(bins[index + 1] * 100).toFixed(0)}%`, count };
  });

  return (
    <div className="summary-panel">
      
      <div className="icon-container">
        <span className="summary-icon iconify" data-icon="material-symbols:info-outline-rounded" data-inline="false"></span>
        <h2>Summary</h2>
      </div>
      
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
        <p><strong>Churn Probability Histogram:</strong></p>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={histogramData} margin={{ top: 20, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="range"/>
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" fill="#C4D600" barSize={50} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SummaryPanel;
