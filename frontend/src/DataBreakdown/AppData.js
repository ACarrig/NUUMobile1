import React, { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const AppData = () => {
  const [chartData1, setChartData1] = useState([]);
  const [chartData2, setChartData2] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:5001/app_usage")
      .then((response) => response.json())
      .then((data) => {
        // Convert dictionary to an array for Recharts
        const formattedData1 = Object.keys(data.hours).map((key) => ({
          name: key,
          value: data.hours[key],
        }));
        const formattedData2 = Object.keys(data.favorite).map((key) => ({
          name: key,
          value: data.favorite[key],
        }));

        setChartData1(formattedData1);
        setChartData2(formattedData2)
      })
      .catch((error) => console.error("Error fetching data:", error));
  }, []);

  // Dynamic y-axis for chart 1 (hours)
  const maxYValue1 = Math.max(...chartData1.map((d) => d.value));
  const minYValue1 = Math.min(...chartData1.map((d) => d.value))
  const yDomain1 = [minYValue1 - 150, maxYValue1 + 100];

  // Dynamic y-axis for chart 2 (# times as most used app)
  const maxYValue2 = Math.max(...chartData2.map((d) => d.value));
  const minYValue2 = Math.min(...chartData2.map((d) => d.value))
  const yDomain2 = [minYValue2 - 250, maxYValue2 + 50];

  return (
    <div>
      <h2 className="centeredHeading">Total App Usage by Hours</h2>
      <ResponsiveContainer width="50%" height={300}>
        <BarChart data={chartData1}>
          <XAxis dataKey="name" />
          <YAxis domain={yDomain1} />
          <Tooltip />
          <Bar dataKey="value" fill="#C4D600" />
        </BarChart>
      </ResponsiveContainer>

      <h2 className="centeredHeading"># of Times as Most Used App</h2>
      <ResponsiveContainer width="50%" height={300}>
        <BarChart data={chartData2}>
          <XAxis dataKey="name" />
          <YAxis domain={yDomain2} />
          <Tooltip />
          <Bar dataKey="value" fill="#C4D600" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AppData;