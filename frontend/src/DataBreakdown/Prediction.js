import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import "./Analysis.css";

const Predictions = () => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');

  const [predictionData, setPredictionData] = useState([]);
  const [hasDeviceNumber, setHasDeviceNumber] = useState(true);

  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchPrediction = async () => {
        try {
          console.log(`Fetching predictions for file: ${selectedFile}, sheet: ${selectedSheet}`);
          const response = await fetch(`http://localhost:5001/predict_data/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          
          if (data.predictions.length > 0) {
            // Check if "Device number" exists in the first row
            setHasDeviceNumber("Device number" in data.predictions[0]);
          }

          setPredictionData(data.predictions);
        } catch (error) {
          console.error("Error fetching prediction:", error);
        }
      };

      fetchPrediction();
    }
  }, [selectedFile, selectedSheet]);

  return (
    <div>
      <h1>Predictions for {selectedFile} - {selectedSheet}</h1>
      <div className="table-container">
        {predictionData.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Row Index</th>
                {hasDeviceNumber && <th>Device Number</th>}
                <th>Churn Prediction</th>
              </tr>
            </thead>
            <tbody>
              {predictionData.map((prediction, index) => (
                <tr key={index}>
                  <td>{prediction["Row Index"]}</td>
                  {hasDeviceNumber && <td>{prediction["Device number"]}</td>}
                  <td>{prediction["Churn Prediction"]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>Loading predictions...</p>
        )}
      </div>
    </div>
  );
};

export default Predictions;
