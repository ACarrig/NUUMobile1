import React, { useEffect, useState } from 'react';
import './Summary.css';

const Summary = ({ selectedFile, selectedSheet, selectedColumn }) => {
  const [aiSummary, setAiSummary] = useState(""); // State for AI summary

  // Fetch the AI response
  useEffect(() => {
    if (selectedFile && selectedSheet) {
      const fetchSummary = async () => {
        try {
          let url = "";

          // Decide which endpoint to call
          if (Array.isArray(selectedColumn) && selectedColumn.length === 2) {
            // ai_summary2 for 2 columns
            const [column1, column2] = selectedColumn;
            url = `http://localhost:5001/ai_summary2?file=${selectedFile}&sheet=${selectedSheet}&column1=${column1}&column2=${column2}`;
          } else {
            // ai_summary for 1 column or no column
            url = `http://localhost:5001/ai_summary?file=${selectedFile}&sheet=${selectedSheet}&column=${selectedColumn}`;
          }

          const response = await fetch(url);
          const data = await response.json();

          if (data && data.summary) {
            setAiSummary(data.summary);
          } else {
            alert('No AI summary received');
          }

        } catch (error) {
          alert(`Error fetching summary: ${error}`);
        }
      };

    fetchSummary();
    }
  }, [selectedFile, selectedSheet, selectedColumn]);

  return (
    <div className="summary-container">
      <h2>Summary</h2>
      <div>
        {aiSummary ? <p>{aiSummary}</p> : <p>Loading summary...</p>}
      </div>
    </div>
  );
};

export default Summary;
