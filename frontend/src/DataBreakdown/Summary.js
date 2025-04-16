import React, { useEffect, useState, useRef } from 'react';
import './Summary.css';

const Summary = ({ selectedFile, selectedSheet, selectedColumn }) => {
  const [aiSummary, setAiSummary] = useState("");
  const abortControllerRef = useRef(null);

  useEffect(() => {
    if (!selectedFile || !selectedSheet) {
      setAiSummary(""); // Clear summary when no file/sheet selected
      return;
    }

    // Cancel any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    const fetchSummary = async () => {
      try {
        let url = "";
        const columnSignature = Array.isArray(selectedColumn) 
          ? selectedColumn.join('|') 
          : selectedColumn;

        if (Array.isArray(selectedColumn) && selectedColumn.length === 2) {
          const [column1, column2] = selectedColumn;
          url = `http://localhost:5001/ai_summary2?file=${selectedFile}&sheet=${selectedSheet}&column1=${column1}&column2=${column2}`;
        } else {
          url = `http://localhost:5001/ai_summary?file=${selectedFile}&sheet=${selectedSheet}&column=${selectedColumn}`;
        }

        const response = await fetch(url, { signal });
        const data = await response.json();

        if (data?.summary) {
          setAiSummary(data.summary);
        } else {
          setAiSummary("No summary available");
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          console.error('Error fetching summary:', error);
          setAiSummary("Error loading summary");
        }
      }
    };

    fetchSummary();

    return () => {
      // Cleanup: abort request when component unmounts or dependencies change
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
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