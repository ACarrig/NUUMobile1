import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import '../Dashboard.css';

const VerificationChart = ({ selectedFile, selectedSheet }) => {
    const [verification, setVerification] = useState([]);    

    useEffect(() => {
        if (selectedFile && selectedSheet) {
            const fetchVerification = async () => {
                try {
                    const response = await fetch(`http://localhost:5001/verification_info/${selectedFile}/${selectedSheet}`);
                    const data = await response.json();
                    if (data.verification) {
                        setVerification(data.verification);
                    } else {
                      console.log('No verification data found');
                    }
                } catch (error) {
                  console.log('Error fetching verification:', error);
                }
            };

            fetchVerification();
        }
    }, [selectedFile, selectedSheet]);

  return (
    <div className="summary-box">
      <h3>Verification</h3>
      <p>Verification of the Defect/Damage or Feedback from the Aftersales Team</p>

      {verification && Object.keys(verification).length ? (
      <div className="summary-graph">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={Object.entries(verification)
            .map(([verification, count]) => ({ verification, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5)}>
            <XAxis dataKey="verification" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#C4D600" />
          </BarChart>
        </ResponsiveContainer>
      </div>
        ) : (
          <p>Loading Verification...</p>
        )}
    </div>
  );
};

export default VerificationChart;