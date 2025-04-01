import React, { useState, useEffect, useRef } from 'react';
import "./Predictions.css";

const PredictionTable = ({ predictionData, hasDeviceNumber }) => {
  // Sorting state for column and sorting order (ascending or descending)
  const [sortColumn, setSortColumn] = useState("Row Index");
  const [sortAscending, setSortAscending] = useState(true);
  
  // Search filter state to filter the predictions by search query
  const [searchQuery, setSearchQuery] = useState("");
  
  // State for showing or hiding the scroll-to-top button based on scroll position
  const [showScrollButton, setShowScrollButton] = useState(false);
  
  // Reference to the table container element for scroll tracking
  const tableContainerRef = useRef(null);

  // Track scroll position and display/hide the scroll button
  useEffect(() => {
    const tableContainer = tableContainerRef.current;
    
    // Event handler to track the scroll position of the table container
    const handleScroll = () => {
      // If the user has scrolled 100px or more, show the scroll-to-top button
      if (tableContainer.scrollTop > 100) {
        setShowScrollButton(true);
      } else {
        setShowScrollButton(false);
      }
    };
    
    // Attach the scroll event listener to the table container
    tableContainer.addEventListener('scroll', handleScroll);
    
    // Clean up the event listener when the component is unmounted or updated
    return () => tableContainer.removeEventListener('scroll', handleScroll);
  }, []);

  // Function to scroll to the top of the table container smoothly
  const scrollToTop = () => {
    if (tableContainerRef.current) {
      tableContainerRef.current.scrollTo({
        top: 0,
        behavior: 'smooth' // Smooth scroll effect
      });
    }
  };

  // Function to toggle the sorting order (ascending or descending) for a column
  const toggleSortOrder = (column) => {
    if (sortColumn === column) {
      // If the clicked column is already the current sort column, toggle the order
      setSortAscending(!sortAscending);
    } else {
      // If a new column is clicked, set the column as the sort column and default to ascending
      setSortColumn(column);
      setSortAscending(true);
    }
  };

  // Filter the prediction data based on the search query
  const filteredPredictionData = predictionData.filter((prediction) => {
    const searchText = searchQuery.toLowerCase();
    return (
      // Filter by row index, device number (if available), churn probability, or churn prediction
      prediction["Row Index"].toString().toLowerCase().includes(searchText) ||
      (hasDeviceNumber && prediction["Device number"] && prediction["Device number"].toString().toLowerCase().includes(searchText)) ||
      prediction["Churn Probability"].toString().toLowerCase().includes(searchText) ||
      prediction["Churn Prediction"].toString().toLowerCase().includes(searchText)
    );
  });

  // Sort the filtered prediction data based on the selected column and sorting order
  const sortedPredictionData = [...filteredPredictionData].sort((a, b) => {
    if (sortColumn === "Row Index") {
      // Sort by row index if the column is "Row Index"
      return sortAscending ? a["Row Index"] - b["Row Index"] : b["Row Index"] - a["Row Index"];
    } else if (sortColumn === "Churn Probability") {
      // Sort by churn probability if the column is "Churn Probability"
      return sortAscending 
        ? a["Churn Probability"] - b["Churn Probability"] 
        : b["Churn Probability"] - a["Churn Probability"];
    }
    return 0;
  });

  return (
    <div className="table-wrapper">
      {/* Search bar to filter the prediction data */}
      <div className="search-bar-container">
        <input
          type="text"
          placeholder="Search..."
          className="search-bar"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)} // Update search query on input change
        />
      </div>

      {/* Table container with scroll tracking */}
      <div className="table-container" ref={tableContainerRef}>
        {filteredPredictionData.length > 0 ? (
          <table>
            <thead>
              <tr>
                {/* Header for Row Index with sorting functionality */}
                <th onClick={() => toggleSortOrder("Row Index")} style={{ cursor: 'pointer' }}>
                  Row Index {sortColumn === "Row Index" ? (sortAscending ? "▲" : "▼") : ""}
                </th>
                {/* Conditionally render Device Number column if the `hasDeviceNumber` prop is true */}
                {hasDeviceNumber && <th>Device Number</th>}
                {/* Header for Churn Probability with sorting functionality */}
                <th onClick={() => toggleSortOrder("Churn Probability")} style={{ cursor: 'pointer' }}>
                  Churn Probability {sortColumn === "Churn Probability" ? (sortAscending ? "▲" : "▼") : ""}
                </th>
                <th>Churn Prediction</th>
              </tr>
            </thead>

            <tbody>
              {/* Render sorted and filtered prediction data */}
              {sortedPredictionData.map((prediction, index) => (
                <tr key={index} className={prediction["Churn Prediction"] === 1 ? "churn-row" : ""}>
                  <td>{prediction["Row Index"]}</td>
                  {/* Conditionally render Device Number if `hasDeviceNumber` is true */}
                  {hasDeviceNumber && <td>{prediction["Device number"]}</td>}
                  {/* Display churn probability as a percentage with 2 decimal places */}
                  <td>{(prediction["Churn Probability"] * 100).toFixed(2)}%</td>
                  <td>{prediction["Churn Prediction"]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p>Loading predictions...</p> // Show loading message if no data is available
        )}
      </div>

      {/* Scroll to top button container */}
      <div className="scroll-button-container">
        {showScrollButton && (
          <button 
            onClick={scrollToTop} // Scroll to the top of the table on button click
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
