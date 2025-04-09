import React, { useEffect, useState } from "react";
import {
    BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";
import { useLocation } from 'react-router-dom';
import "./Analysis.css";

const MonthlySales = () => {
    const [monthlySaleTotals, setMonthlySaleTotals] = useState([]);
    const [monthlyDeviceSales, setMonthlyDeviceSales] = useState([]);
    const [monthlyRetention, setMonthlyRetention] = useState([]);

    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

    // Fetch Monthly Sales
    useEffect(() => {
        fetch(`http://localhost:5001/get_monthly_sales/${selectedFile}/${selectedSheet}`)
        .then((response) => response.json())
        .then(json => {
            const barChartData = Object.entries(json.monthlySales).map(([month, value]) => ({
                month,
                value: value || 0
            }));
    
            // Sort by the numerical representation of the months
            barChartData.sort((a, b) => monthToNumber(a.month) - monthToNumber(b.month));
    
            setMonthlySaleTotals(barChartData);
        })
    }, [selectedFile, selectedSheet]);

    // Fetch Monthly Sales by Device (Grouped Bar Chart)
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch(`http://127.0.0.1:5001/get_monthly_model_sales/${selectedFile}/${selectedSheet}`);
                const data = await response.json();
                setMonthlyDeviceSales(data.modelSales);
            } catch (error) {
                console.error('Error fetching sales data:', error);
            }
        };
        fetchData();
    }, [selectedFile, selectedSheet]);

    // Fetch Monthly Retention (Bar Chart)
    useEffect(() => {
        fetch(`http://127.0.0.1:5001/get_monthly_retainment/${selectedFile}/${selectedSheet}`)
        .then((response) => response.json())
        .then(json => {
            const barChartData = Object.entries(json.modelRetention).map(([month, value]) => ({
                month,
                value: value || 0
            }));
    
            // Sort by the numerical representation of the months
            barChartData.sort((a, b) => monthToNumber(a.month) - monthToNumber(b.month));
    
            setMonthlyRetention(barChartData);
        })
    }, [selectedFile, selectedSheet]);

    const monthToNumber = (month) => {
        const monthMap = {
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
            "November": 11
        };
        return monthMap[month] || 0;
    };

    const processChartTwoData = () => {
        if (!monthlyDeviceSales) return [];
    
        const months = Object.keys(monthlyDeviceSales);
        const allModels = [
          ...new Set(months.flatMap((month) => Object.keys(monthlyDeviceSales[month]))),
        ];
    
        const sortedMonths = months.sort((a, b) => monthToNumber(a) - monthToNumber(b));  // Sort months numerically
    
        return sortedMonths.map((month) => {
          const monthData = { month };
          allModels.forEach((model) => {
            monthData[model] = monthlyDeviceSales[month][model] || 0;
          });
          return monthData;
        });
    };
    
    const chartTwoData = processChartTwoData();

    return (
        <div className="content">

            <div><h1>Monthly Sales Overview</h1></div>

            <div>
                <h3>Sales Over Time</h3>
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={monthlySaleTotals} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Line 
                            dataKey="value"
                            stroke="#C4D600"
                            strokeWidth={4}
                            dot={{ r: 4 }}
                            activeDot={{ r: 6 }} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div>
                <h3>Sales by Device Per Month</h3>
                {monthlyDeviceSales ? (
                    <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={chartTwoData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        {Object.keys(chartTwoData[0] || {}).filter((key) => key !== 'month').map((model) => (
                        <Bar key={model} dataKey={model} name={model} fill="#C4D600" />
                        ))}
                    </BarChart>
                    </ResponsiveContainer>
                ) : (
                    <p>Loading...</p>
                )}
            </div>

            <div>
                <h3>Average Device Retention</h3>
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={monthlyRetention} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Line 
                            dataKey="value"
                            stroke="#3d3d3b"
                            strokeWidth={4}
                            dot={{ r: 4 }}
                            activeDot={{ r: 6 }} />
                    </LineChart>
                </ResponsiveContainer>
            </div>

        </div>
    );
};

export default MonthlySales;
