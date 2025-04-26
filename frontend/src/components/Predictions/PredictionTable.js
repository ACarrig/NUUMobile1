import React, { useState, useEffect, useRef } from 'react';
import "./Predictions.css";

const PredictionTable = ({ selectedFile, selectedSheet, selectedModel }) => {
  const [predictionData, setPredictionData] = useState([]);
  const [hasDeviceNumber, setHasDeviceNumber] = useState(false);
  const [sortColumn, setSortColumn] = useState("Row Index");
  const [sortAscending, setSortAscending] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [showScrollButton, setShowScrollButton] = useState(false);
  const tableContainerRef = useRef(null);

  useEffect(() => {
    const fetchPredictionData = async () => {
      if (!selectedFile || !selectedSheet || !selectedModel) return;
      setPredictionData([]);
  
      const modelToEndpoint = {
        ensemble: 'em',
        xgb: 'xgb',
        nn: 'nn'
      };
  
      try {
        const endpointPrefix = modelToEndpoint[selectedModel] || 'em'; // fallback if needed
        const predictionResponse = await fetch(
          `http://localhost:5001/${endpointPrefix}_predict_data/${selectedFile}/${selectedSheet}`
        );
        const predictionJson = await predictionResponse.json();
        if (predictionJson.predictions) {
          setPredictionData(predictionJson.predictions);
          setHasDeviceNumber("Device number" in predictionJson.predictions[0]);
        } else {
          setPredictionData([]);
          setHasDeviceNumber(false);
        }
      } catch (error) {
        console.error("Error fetching prediction data:", error);
        setPredictionData([]);
        setHasDeviceNumber(false);
      }
    };
  
    fetchPredictionData();
  }, [selectedFile, selectedSheet, selectedModel]);
  
  useEffect(() => {
    const tableContainer = tableContainerRef.current;
    const handleScroll = () => {
      setShowScrollButton(tableContainer.scrollTop > 100);
    };

    if (tableContainer) {
      tableContainer.addEventListener('scroll', handleScroll);
    }

    return () => {
      if (tableContainer) {
        tableContainer.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  const scrollToTop = () => {
    tableContainerRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

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
      (hasDeviceNumber && prediction["Device number"]?.toString().toLowerCase().includes(searchText)) ||
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
      <div className="search-bar-container">
        <input
          type="text"
          placeholder="Search..."
          className="search-bar"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      <div className="table-container" ref={tableContainerRef}>
        {predictionData.length > 0 ? (
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

      {showScrollButton && (
        <div className="scroll-button-container">
          <button onClick={scrollToTop} className="scroll-to-top" aria-label="Scroll to top of table">
            ↑
          </button>
        </div>
      )}
    </div>
  );
};

export default PredictionTable;
