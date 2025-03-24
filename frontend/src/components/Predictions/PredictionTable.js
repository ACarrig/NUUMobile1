import React, { useState } from 'react';
import "./Predictions.css";

const PredictionTable = ({ predictionData, hasDeviceNumber }) => {
  // Sorting state
  const [sortColumn, setSortColumn] = useState("Row Index"); // Default sort column
  const [sortAscending, setSortAscending] = useState(true); // Default sort order

  // Search filter state
  const [searchQuery, setSearchQuery] = useState("");

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

  return (
    <div className="table-wrapper">
      {/* Search bar */}
      <div className="search-bar-container">
        <input
          type="text"
          placeholder="Search..."
          className="search-bar"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
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
  );
};

export default PredictionTable;
