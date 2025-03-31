import React, { useEffect, useState } from 'react';
import './Dashboard.css';

const MonthlySalesChart = ({openwindow, selectedFile, selectedSheet}) => {
    const [monthlySaleTotals, setMonthlySaleTotals] = useState({});

    // Fetch sales by month for this component (dashboard preview for monthlySales page)
    useEffect(() => {
        const fetchMonthlySales = async () => {
            try {
                const response = await fetch(`http://localhost:5001/get_monthly_sales/${selectedFile}/${selectedSheet}`);
                const data = await response.json();

                if (data) {
                    setMonthlySaleTotals(data);
                }
            } catch (error) {
                console.error('Error getting monthly sales')
            }
        };
        fetchMonthlySales();
    }, [selectedFile, selectedSheet]);

    return(
        <div className="summary-box">
            <h3>Top 5 Most Used Apps</h3>
            {top5Apps && Object.keys(top5Apps).length ? (
                <ol>
                {Object.entries(top5Apps)
                    .sort((a, b) => b[1] - a[1])
                    .map(([app, usage], index) => (
                    <li key={index}>{app}: {usage} hrs</li>
                    ))}
                </ol>
            ) : (
                <p>Loading top 5 apps...</p>
            )}
            <button onClick={() => openWindow('/monthlysales')}>View App Data</button>
        </div>
    );

};

export default MonthlySalesChart;