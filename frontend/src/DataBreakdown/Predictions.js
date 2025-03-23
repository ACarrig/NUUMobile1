import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom'; 
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import FileSelector from '../components/FileSelector';
import SheetSelector from '../components/SheetSelector';
import "./Predictions.css";

const Predictions = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const queryParams = new URLSearchParams(location.search);
  const initialSelectedFile = queryParams.get('file') || '';
  const initialSelectedSheet = queryParams.get('sheet') || '';

  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(initialSelectedFile);
  const [sheets, setSheets] = useState([]);
  const [selectedSheet, setSelectedSheet] = useState(initialSelectedSheet);
  const [predictionData, setPredictionData] = useState([]);
  const [hasDeviceNumber, setHasDeviceNumber] = useState(true);
  const [featureImportances, setFeatureImportances] = useState([]);

  // Sorting state
  const [sortColumn, setSortColumn] = useState("Row Index"); // Default sort column
  const [sortAscending, setSortAscending] = useState(true); // Default sort order

  // Search filter state
  const [searchQuery, setSearchQuery] = useState("");

  // Fetch files, sheets, and predictions
  useEffect(() => {
    const fetchData = async () => {
      try {
        const fileResponse = await fetch('http://localhost:5001/get_files');
        const fileData = await fileResponse.json();
        if (fileData.files) {
          setFiles(fileData.files);
        }

        if (selectedFile !== '') {
          const sheetResponse = await fetch(`http://localhost:5001/get_sheets/${selectedFile}`);
          const sheetData = await sheetResponse.json();
          if (sheetData.sheets) {
            setSheets(sheetData.sheets);
          }

          if (selectedSheet !== '') {
            const predictionResponse = await fetch(`http://localhost:5001/predict_data/${selectedFile}/${selectedSheet}`);
            const predictionData = await predictionResponse.json();
            if (predictionData.predictions) {
              setPredictionData(predictionData.predictions);
              setHasDeviceNumber("Device number" in predictionData.predictions[0]);
            }
          }
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, [selectedFile, selectedSheet]);

  // Handle file selection change
  const handleFileSelectChange = (event) => {
    const file = event.target.value;
    setSelectedFile(file);
    setSelectedSheet('');
    navigate(`?file=${file}&sheet=`);
  };

  // Handle sheet selection change
  const handleSheetSelectChange = (event) => {
    const sheet = event.target.value;
    setSelectedSheet(sheet);
    navigate(`?file=${selectedFile}&sheet=${sheet}`);
  };

  // Toggle sorting order for a given column
  const toggleSortOrder = (column) => {
    if (sortColumn === column) {
      setSortAscending(!sortAscending);
    } else {
      setSortColumn(column);
      setSortAscending(true); // Default to ascending when switching columns
    }
  };

  // Filter predictions based on search query
  const filteredPredictionData = predictionData.filter((prediction) => {
    const searchText = searchQuery.toLowerCase();
    return (
      prediction["Row Index"].toString().toLowerCase().includes(searchText) ||
      (hasDeviceNumber && prediction["Device number"] && prediction["Device number"].toString().toLowerCase().includes(searchText)) ||
      prediction["Churn Probability"].toString().toLowerCase().includes(searchText) ||
      prediction["Churn Prediction"].toString().toLowerCase().includes(searchText)
    );
  });

  // Sort the filtered predictions
  const sortedPredictionData = [...filteredPredictionData].sort((a, b) => {
    if (sortColumn === "Row Index") {
      return sortAscending ? a["Row Index"] - b["Row Index"] : b["Row Index"] - a["Row Index"];
    } else if (sortColumn === "Churn Probability") {
      return sortAscending 
        ? a["Churn Probability"] - b["Churn Probability"] 
        : b["Churn Probability"] - a["Churn Probability"];
    }
    return 0;
  });

  // Function to format unmapped feature names
  const formatFeatureName = (feature) => {
      return feature
          .replace(/_/g, " ") // Replace underscores with spaces
          .replace(/-/g, " ") // Replace dashes with spaces
          .replace(/\b\w/g, char => char.toUpperCase()); // Capitalize each word
  };
  
  useEffect(() => {
      if (selectedFile && selectedSheet) {
          const fetchFeatureImportances = async () => {
              try {
                  console.log(`Fetching model frequency for file: ${selectedFile}, sheet: ${selectedSheet}`);
                  const response = await fetch(`http://localhost:5001/get_features`);
                  const data = await response.json();
                  if (data.features) {
                      // Apply user-friendly formatting
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

  return (
    <div className="predictions-container">
      <h1>Predictions for {selectedFile} - {selectedSheet}</h1>

      <div className="dropdown-container">
        <FileSelector files={files} selectedFile={selectedFile} onFileChange={handleFileSelectChange} />
        {selectedFile && (
          <SheetSelector sheets={sheets} selectedSheet={selectedSheet} onSheetChange={handleSheetSelectChange} />
        )}
      </div>

      {selectedFile && selectedSheet && (
        <div className="content-container">
          <div className="summary-panel">
            <h2>Summary</h2>
            <p><strong>Total Rows:</strong> {filteredPredictionData.length}</p>

            <div className="summary-item">
              <p><strong>Churn Predictions:</strong></p>
              <ul>
                <li><strong>Churn (1):</strong> {filteredPredictionData.filter(p => p["Churn Prediction"] === 1).length} </li>
                <li><strong>Not Churn (0):</strong> {filteredPredictionData.filter(p => p["Churn Prediction"] === 0).length}</li>
                <li><strong>Churn Rate:</strong> {((filteredPredictionData.filter(p => p["Churn Prediction"] === 1).length / filteredPredictionData.length) * 100).toFixed(2)}%</li>
              </ul>
            </div>

            {/* Churn Probability Stats */}
            <div className="summary-item">
              <p><strong>Churn Probability Stats:</strong></p>
              <ul>
                <li><strong>Average Probability:</strong> {filteredPredictionData.length > 0 ? (filteredPredictionData.reduce((acc, p) => acc + p["Churn Probability"], 0) / filteredPredictionData.length * 100).toFixed(2) + "%" : "N/A"}</li>
                <li><strong>Max Probability:</strong> {filteredPredictionData.length > 0 ? (Math.max(...filteredPredictionData.map(p => p["Churn Probability"])) * 100).toFixed(2) + "%" : "N/A"}</li>
                <li><strong>Min Probability:</strong> {filteredPredictionData.length > 0 ? (Math.min(...filteredPredictionData.map(p => p["Churn Probability"])) * 100).toFixed(2) + "%" : "N/A"}</li>
              </ul>
            </div>

            {/* Churn Probability Distribution */}
            <div className="summary-item">
              <p><strong>Churn Probability Distribution:</strong></p>
              <ul>
                <li>0-20%: {filteredPredictionData.filter(p => p["Churn Probability"] <= 0.2).length}</li>
                <li>20-40%: {filteredPredictionData.filter(p => p["Churn Probability"] > 0.2 && p["Churn Probability"] <= 0.4).length}</li>
                <li>40-60%: {filteredPredictionData.filter(p => p["Churn Probability"] > 0.4 && p["Churn Probability"] <= 0.6).length}</li>
                <li>60-80%: {filteredPredictionData.filter(p => p["Churn Probability"] > 0.6 && p["Churn Probability"] <= 0.8).length}</li>
                <li>80-100%: {filteredPredictionData.filter(p => p["Churn Probability"] > 0.8).length}</li>
              </ul>
            </div>
          </div>

          <div className="table-wrapper">
            {/* Search bar */}
            <div className="search-bar-container">
              <input
                type="text"
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="search-bar"
              />
            </div>

            <div className="table-container">
              {filteredPredictionData.length > 0 ? (
                <table>
                  <thead>
                    <tr>
                      <th onClick={() => toggleSortOrder("Row Index")} style={{ cursor: 'pointer' }}>
                        Row Index {sortColumn === "Row Index" ? (sortAscending ? "▲" : "▼") : ""}
                      </th>
                      {hasDeviceNumber && <th>Device Number</th>}
                      <th onClick={() => toggleSortOrder("Churn Probability")} style={{ cursor: 'pointer' }}>
                        Churn Probability {sortColumn === "Churn Probability" ? (sortAscending ? "▲" : "▼") : ""}
                      </th>
                      <th>Churn Prediction</th>
                    </tr>
                  </thead>

                  <tbody>
                    {sortedPredictionData.map((prediction, index) => (
                      <tr key={index} className={prediction["Churn Prediction"] === 1 ? "churn-row" : ""}>
                        <td>{prediction["Row Index"]}</td>
                        {hasDeviceNumber && <td>{prediction["Device number"]}</td>}
                        <td>{(prediction["Churn Probability"] * 100).toFixed(2)}%</td>
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

        </div>
      )}

      <div>
        {/* Feature Importance Graph */}
        {featureImportances.length > 0 && (
            <div className="feature-importance-container">
              <h2>Feature Importances</h2>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={featureImportances} layout="vertical">
                  <XAxis type="number" />
                  <YAxis dataKey="Feature" type="category" width={150} />
                  <Tooltip />
                  <Bar dataKey="Importance" fill="#C4D600" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
      </div>

    </div>

  );  
};

export default Predictions;
