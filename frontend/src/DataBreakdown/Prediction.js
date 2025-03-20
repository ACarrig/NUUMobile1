import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import "./Analysis.css";

const Predictions = () => {
  const location = useLocation();
  const queryParams = new URLSearchParams(location.search);
  const selectedFile = queryParams.get('file');
  const selectedSheet = queryParams.get('sheet');

  const [sheetData, setSheetData] = useState([]);

  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchData = async () => {
        try {
          console.log(`Fetching data for file: ${selectedFile}, sheet: ${selectedSheet}`);
          const response = await fetch(`http://localhost:5001/predict_data/${selectedFile}/${selectedSheet}`);
          const data = await response.json();
          setSheetData(data.predictions);
        } catch (error) {
          console.error("Error fetching sheet data:", error);
        }
      };

      fetchData();
    }
  }, [selectedFile, selectedSheet]);

  return (
    <div>
      <h1>Predictions for {selectedFile} - {selectedSheet}</h1>
      <div className="table-container">
        {sheetData.length > 0 ? (
          <table>
            <thead>
              <tr>
                <th>Device Number</th>
                <th>Churn Prediction</th>
              </tr>
            </thead>
            <tbody>
              {sheetData.map((prediction, index) => (
                <tr key={index}>
                  <td>{prediction['Device number']}</td>
                  <td>{prediction['Churn Prediction']}</td>
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
