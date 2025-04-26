import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import '../Dashboard.css';

const ResPartyChart = ({ selectedFile, selectedSheet }) => {
    const [resParty, setResParty] = useState([]);    

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchResParty = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/resparty_info/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();
                    if (data.responsible_party) {
                        setResParty(data.responsible_party);
                    } else {
                      console.log('No responsible party found');
                    }
                } catch (error) {
                  console.log('Error fetching responsible party:', error);
                }
            };

            fetchResParty();
        }
    }, [selectedFile, selectedSheet]);

  return (
    <div className="summary-box">
      <h3>Returns Feedback</h3>
      <p>The reason why the device is returned</p>

      {resParty && Object.keys(resParty).length ? (
      <div className="summary-graph">
        <ResponsiveContainer width="100%" height={200}>
            <BarChart data={Object.entries(resParty)
            .map(([resParty, count]) => ({ resParty, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5)}>
            <XAxis dataKey="resParty" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#C4D600" />
          </BarChart>
        </ResponsiveContainer>
      </div>
        ) : (
          <p>Loading Responsible Party...</p>
        )}
        </div>
  );
};

export default ResPartyChart;
