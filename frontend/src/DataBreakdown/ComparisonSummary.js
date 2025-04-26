import React, { useEffect, useState, useRef } from 'react';
import './Summary.css';

const Summary = ({ selectedFile, selectedSheet, selectedColumns }) => {
  const [aiSummary, setAiSummary] = useState("");
  const lastRequestRef = useRef(""); // Store last request signature

  useEffect(() => {
    if (!selectedFile || !selectedSheet || !selectedColumns || selectedColumns.length === 0) return;

    const columnSignature = selectedColumns.join('|'); // Join columns for signature

    const requestSignature = `${selectedFile}__${selectedSheet}__${columnSignature}`;

    if (requestSignature === lastRequestRef.current) return; // Avoid unnecessary repeated requests
    lastRequestRef.current = requestSignature; // Save this as the last request

    const fetchSummary = async () => {
      try {
        let url = `http://localhost:5001/comparison_summary?file=${selectedFile}&sheet=${selectedSheet}`;

        // Append selected columns dynamically to the URL
        selectedColumns.forEach((column, index) => {
          url += `&column=${column}`;
        });

        const response = await fetch(url);
        const data = await response.json();

        if (data && data.summary) {
          setAiSummary(data.summary);
        } else {
          console.log('No AI summary received');
        }

      } catch (error) {
        console.log(`Error fetching summary: ${error}`);
      }
    };

    fetchSummary();
  }, [selectedFile, selectedSheet, selectedColumns]);

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
