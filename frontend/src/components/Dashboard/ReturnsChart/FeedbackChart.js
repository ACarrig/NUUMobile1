import React, { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip} from 'recharts';
import '../Dashboard.css';

const FeedbackChart = ({ openWindow, selectedFile, selectedSheet }) => {
  const [feedback, setFeedback] = useState([]);

  useEffect(() => {
    if (selectedFile && selectedSheet) {
    const fetchFeedback = async () => {
      try {
      const response = await fetch(`http://localhost:5001/feedback_info/${selectedFile}/${selectedSheet}`);
      const data = await response.json();
      if (data.feedback) {
        setFeedback(data.feedback);
      } else {
        console.log('No feedback data found');
      }
      } catch (error) {
        console.log('Error fetching feedback:', error);
      }
    };

    fetchFeedback();
    }
  }, [selectedFile, selectedSheet]);

  return (
    <div className="summary-box">
      <h3>Returns Feedback</h3>
      <p>Feedback about Returned Devices from Customer</p>

      {feedback && Object.keys(feedback).length ? (
      <div className="summary-graph">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={Object.entries(feedback)
            .map(([feedback, count]) => ({ feedback, count }))
            .sort((a, b) => b.count - a.count)
            .slice(0, 5)}>
            <XAxis dataKey="feedback" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#C4D600" />
          </BarChart>
        </ResponsiveContainer>
      </div>
        ) : (
          <p>Loading Feedback...</p>
        )}
    </div>
  );
};

export default FeedbackChart;
