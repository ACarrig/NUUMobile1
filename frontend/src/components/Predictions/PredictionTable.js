import React, { useState, useEffect, useRef } from 'react';
import "./Predictions.css";

const PredictionTable = ({ predictionData, hasDeviceNumber }) => {
  // Sorting state
  const [sortColumn, setSortColumn] = useState("Row Index");
  const [sortAscending, setSortAscending] = useState(true);
  
  // Search filter state
  const [searchQuery, setSearchQuery] = useState("");
  
  // Scroll state
  const [showScrollButton, setShowScrollButton] = useState(false);
  const tableContainerRef = useRef(null);

  // Check scroll position within the table container
  useEffect(() => {
    const tableContainer = tableContainerRef.current;
    
    const handleScroll = () => {
      if (tableContainer.scrollTop > 100) {
        setShowScrollButton(true);
      } else {
        setShowScrollButton(false);
      }
    };
    
    tableContainer.addEventListener('scroll', handleScroll);
    return () => tableContainer.removeEventListener('scroll', handleScroll);
  }, []);

  // Scroll to top of the table
  const scrollToTop = () => {
    if (tableContainerRef.current) {
      tableContainerRef.current.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }
  };

  // Rest of your component code remains the same...
  const toggleSortOrder = (column) => {
    if (sortColumn === column) {
      setSortAscending(!sortAscending);
    } else {
      setSortColumn(column);
      setSortAscending(true);
    }
  };

  const filteredPredictionData = predictionData.filter((prediction) => {
    const searchText = searchQuery.toLowerCase();
    return (
      prediction["Row Index"].toString().toLowerCase().includes(searchText) ||
      (hasDeviceNumber && prediction["Device number"] && prediction["Device number"].toString().toLowerCase().includes(searchText)) ||
      prediction["Churn Probability"].toString().toLowerCase().includes(searchText) ||
      prediction["Churn Prediction"].toString().toLowerCase().includes(searchText)
    );
  });

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

      {/* Table container with scroll tracking */}
      <div className="table-container" ref={tableContainerRef}>
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

      {/* Scroll to top button container */}
      <div className="scroll-button-container">
        {showScrollButton && (
          <button 
            onClick={scrollToTop}
            className="scroll-to-top"
            aria-label="Scroll to top of table"
          >
            ↑
          </button>
        )}
      </div>

    </div>
  );
};

export default PredictionTable;