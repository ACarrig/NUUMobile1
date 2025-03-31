import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const MonthlySales = () => {
    const [monthlySaleTotals, setMonthlySaleTotals] = useState({});
    const [monthlyDeviceSales, setMonthlyDeviceSales] = useState({});
    const [monthlyRetention, setMonthlyRetention] = useState({});

    // Fetch monthly sales
    useEffect(() => {
        const fetchMonthlySales = async () => {
            try {
                const response = await fetch(`http://localhost:5001/get_monthly_sales/${selectedFile}/${selectedSheet}`);
                const data = await response.json();

                if (data) {
                    setMonthlySaleTotals(data);
                }
            } catch (error) {
                console.error('Error getting monthly sales');
            }
        };
        fetchMonthlySales();
    });

    // Fetch monthly sales by device
    useEffect(() => {
        const fetchMonthlyDeviceSales = async () => {
            try {
                const response = await fetch(`http://localhost:5001/get_monthly_model_sales/${selectedFile}/${selectedSheet}`);
                const data = await response.json();

                if (data) {
                    setMonthlyDeviceSales(data);
                }
            } catch (error) {
                console.error('Error getting monthly sales by device');
            }
        };
        fetchMonthlyDeviceSales();
    });

    // Fetch avg monthly retention of devices
    useEffect(() => {
        const fetchAvgRetainment = async () => {
            try {
                const response = await fetch(`http://localhost:5001/get_monthly_retainment/${selectedFile}/${selectedSheet}`);
                const data = response.json();

                if (data) {
                    setMonthlyRetention(data);
                }

            } catch (error) {
                console.error('Could not get avg device retainment');
            }
        };
        fetchAvgRetainment();
    });

    return(
        <div>
            
        </div>
    );
};

export default MonthlySales;