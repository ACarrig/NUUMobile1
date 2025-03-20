import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip, ResponsiveContainer, PieChart, Pie, Cell, Tooltip, Legend } from 'recharts'; 
import "./Analysis.css";

const ParamCorr = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    const [correlation, setCorrelation] = useState({});
    const [aiSummary, setAiSummary] = useState("");

    useEffect(() => {
            if (selectedFile && selectedSheet) {
                const fetchCorrData = async () => {
                  try {
                    const response = await fetch(`http://localhost:5001/param_churn_correlation/${selectedFile}`);
                    const data = await response.json();
                    if (data.corr) {
                      setCorrelation(data.corr);
                    } else {
                      alert('No data found');
                    }
                  } catch (error) {
                    alert('Error fetching correlation data:', error);
                  }
                };
    
                fetchCorrData();
            }
        }, [selectedFile, selectedSheet]);

    return (
        //
    );
};