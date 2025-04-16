import React, { useEffect, useState, useRef } from 'react';
import './Summary.css';

const Summary = ({ selectedFile, selectedSheet, selectedColumn }) => {
  const [aiSummary, setAiSummary] = useState("");
  const lastRequestRef = useRef(""); // Store last request signature

  useEffect(() => {
    if (!selectedFile || !selectedSheet) return;

    const columnSignature = Array.isArray(selectedColumn) 
      ? selectedColumn.join('|') 
      : selectedColumn;

    const requestSignature = `${selectedFile}__${selectedSheet}__${columnSignature}`;

    if (requestSignature === lastRequestRef.current) return; // Already fetched
    lastRequestRef.current = requestSignature; // Save this as the last request

    const fetchSummary = async () => {
      try {
        let url = "";

        if (Array.isArray(selectedColumn) && selectedColumn.length === 2) {
          const [column1, column2] = selectedColumn;
          url = `http://localhost:5001/ai_summary2?file=${selectedFile}&sheet=${selectedSheet}&column1=${column1}&column2=${column2}`;
        } else {
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

  }, [selectedFile, selectedSheet, selectedColumn]);

  return (
    <div className="summary-container">
      <h2>AI Summary</h2>
      <div>
        {aiSummary ? <p>{aiSummary}</p> : <p>Loading summary...</p>}
      </div>
    </div>
  );
};

export default Summary;
