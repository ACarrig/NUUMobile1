import React, { useEffect, useState } from 'react';
import '../Dashboard.css';

const MonthlySalesChart = ({ openWindow, selectedFile, selectedSheet }) => {
    const [monthlySaleTotals, setMonthlySaleTotals] = useState({});

    // Fetch sales by month for this component (dashboard preview for monthlySales page)
    useEffect(() => {
        const fetchMonthlySales = async () => {
            try {
                const response = await fetch(`http://localhost:5001/get_monthly_sales/${selectedFile}/${selectedSheet}`);
                const data = await response.json();

                if (data.monthlySales) {
                    setMonthlySaleTotals(data.monthlySales);
                }
            } catch (error) {
                console.error('Error getting monthly sales:', error);
            }
        };
        fetchMonthlySales();
    }, [selectedFile, selectedSheet]);

    // only display non-0 months in the preview
    const allowedMonths = ["July", "August", "September", "October", "November"];

    return (
        <div className="summary-box">
            <h3>Sales by Month</h3>
            <ul>
                {Object.entries(monthlySaleTotals)
                    .filter(([month]) => allowedMonths.includes(month))
                    .map(([month, sales]) => (
                        <li key={month}>{month}: {sales}</li>
                    ))
                }
            </ul>
            <button onClick={() => openWindow(`/monthlysales?file=${selectedFile}&sheet=${selectedSheet}`)}>View Monthly Sales Data</button>
        </div>
    );
};

export default MonthlySalesChart;
