import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import "./Predictions.css";

const SummaryPanel = ({ selectedFile, selectedSheet, selectedModel }) => {
  const [predictionData, setPredictionData] = useState(null);

  useEffect(() => {
    const fetchPredictionData = async () => {
      if (!selectedFile || !selectedSheet || !selectedModel) {
        setPredictionData(null);
        return;
      }

      setPredictionData(null);

      const modelToEndpoint = {
        ensemble: "em",
        xgb: 'xgb',
        nn: "nn",
      };

      try {
        const endpointPrefix = modelToEndpoint[selectedModel] || "em"; // fallback to ensemble

        const predictionResponse = await fetch(
          `http://localhost:5001/${endpointPrefix}_predict_data/${selectedFile}/${selectedSheet}`
        );
        const predictionJson = await predictionResponse.json();

        if (predictionJson.predictions) {
          setPredictionData(predictionJson.predictions);
        } else {
          setPredictionData([]);
        }
      } catch (error) {
        console.error("Error fetching prediction data:", error);
        setPredictionData([]);
      }
    };

    fetchPredictionData();
  }, [selectedFile, selectedSheet, selectedModel]);

  if (!predictionData || predictionData.length === 0) {
    return (
      <div className="summary-panel">
        <p>Loading summary of predictions...</p>
      </div>
    );
  }

  const churnCount = predictionData.filter((p) => p["Churn Prediction"] === 1).length;
  const notChurnCount = predictionData.filter((p) => p["Churn Prediction"] === 0).length;
  const churnRate = ((churnCount / predictionData.length) * 100).toFixed(2);

  const avgProbability = (
    (predictionData.reduce((acc, p) => acc + p["Churn Probability"], 0) / predictionData.length) *
    100
  ).toFixed(2);
  const maxProbability = (Math.max(...predictionData.map((p) => p["Churn Probability"])) * 100).toFixed(2);
  const minProbability = (Math.min(...predictionData.map((p) => p["Churn Probability"])) * 100).toFixed(2);

  // Define bins dynamically for histogram
  const bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]; // 5 bins
  const histogramData = bins.slice(0, -1).map((bin, index) => {
    const count = predictionData.filter(
      (p) => p["Churn Probability"] >= bin && p["Churn Probability"] < bins[index + 1]
    ).length;
    return { range: `${(bin * 100).toFixed(0)}-${(bins[index + 1] * 100).toFixed(0)}%`, count };
  });

  const handleDownload = () => {
    const modelToEndpoint = {
      ensemble: "em",
      mlp: "mlp",
      nn: "nn",
    };
    const endpointPrefix = modelToEndpoint[selectedModel] || "em";
    const url = `http://localhost:5001/${endpointPrefix}_download_data/${selectedFile}/${selectedSheet}`;
    window.open(url, "_blank");
  };

  return (
    <div>
      <div className="summary-panel">
        <div className="icon-container">
          <span
            className="summary-icon iconify"
            data-icon="material-symbols:info-outline-rounded"
            data-inline="false"
          ></span>
          <h2>Summary</h2>
        </div>

        <p>
          <strong>Total Rows:</strong> {predictionData.length}
        </p>

        <div className="summary-item">
          <p>
            <strong>Churn Predictions:</strong>
          </p>
          <ul>
            <li>
              <strong>Churn (1):</strong> {churnCount}
            </li>
            <li>
              <strong>Not Churn (0):</strong> {notChurnCount}
            </li>
            <li>
              <strong>Churn Rate:</strong> {churnRate}%
            </li>
          </ul>
        </div>

        <div className="summary-item">
          <p>
            <strong>Churn Probability Stats:</strong>
          </p>
          <ul>
            <li>
              <strong>Average Probability:</strong> {avgProbability}%
            </li>
            <li>
              <strong>Max Probability:</strong> {maxProbability}%
            </li>
            <li>
              <strong>Min Probability:</strong> {minProbability}%
            </li>
          </ul>
        </div>

        <div className="summary-item">
          <p>
            <strong>Churn Probability Histogram:</strong>
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={histogramData} margin={{ top: 20, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="range" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="count" fill="#C4D600" barSize={50} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {selectedFile && selectedSheet && (
        <div className="download-button-container">
          <button onClick={handleDownload} className="download-button">
            Download Predictions CSV
          </button>
        </div>
      )}
    </div>
  );
};

export default SummaryPanel;
