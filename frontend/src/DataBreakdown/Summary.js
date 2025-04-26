import React, { useEffect, useState, useRef, useCallback } from 'react';
import './Summary.css';

const Summary = ({ selectedFile, selectedSheet, selectedColumn }) => {
  const [aiSummary, setAiSummary] = useState("");
  const [isRefreshing, setIsRefreshing] = useState(false);
  const lastRequestRef = useRef("");

  // Memoize fetchSummary with useCallback
  const fetchSummary = useCallback(async (isManualRefresh = false) => {
    if (!selectedFile || !selectedSheet) return;
    setIsRefreshing(true);
    setAiSummary("");
    
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
        console.log('No AI summary received');
      }

    } catch (error) {
      console.log(`Error fetching summary: ${error}`);
    } finally {
      setIsRefreshing(false);
    }
  }, [selectedFile, selectedSheet, selectedColumn]); // Add dependencies here

  useEffect(() => {
    const columnSignature = Array.isArray(selectedColumn) 
      ? selectedColumn.join('|') 
      : selectedColumn;

    const requestSignature = `${selectedFile}__${selectedSheet}__${columnSignature}`;

    if (requestSignature === lastRequestRef.current) return;
    lastRequestRef.current = requestSignature;

    fetchSummary(false); // Pass false to indicate this is not a manual refresh
  }, [selectedFile, selectedSheet, selectedColumn, fetchSummary]); // Add fetchSummary to dependencies

  return (
    <div className="summary-container">
      <div className="summary-header">
        <h2>AI Summary</h2>
        <button 
          className="refresh-button" 
          onClick={() => fetchSummary(true)}
          disabled={isRefreshing}>
          <span 
            className={`summary-icon iconify`} 
            data-icon="material-symbols:refresh-rounded" 
            data-inline="false"></span>
        </button>
      </div>

      <div>
        {aiSummary ? (
          <p>{aiSummary}</p>
        ) : (
          <p>{isRefreshing ? 'Loading summary...' : 'No summary available'}</p>
        )}
      </div>
    </div>
  );
};

export default Summary;