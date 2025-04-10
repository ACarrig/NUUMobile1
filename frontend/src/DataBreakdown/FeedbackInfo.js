import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import "./Analysis.css";

const FeedbackInfo = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const selectedFile = queryParams.get('file');
    const selectedSheet = queryParams.get('sheet');

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
                            alert('No feedback data found');
                        }
                    } catch (error) {
                    alert('Error fetching feedback:', error);
                }
            };

        fetchFeedback();
        }
    }, [selectedFile, selectedSheet]);

    return (
        <div className='content'>
            <h1>Feedback Info for {selectedFile} - {selectedSheet}</h1>
            {feedback && Object.keys(feedback).length ? (
            <div className="graph-container">
                <div className='chart'>
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={Object.entries(feedback)
                    .map(([feedback, count]) => ({ feedback, count }))
                    .sort((a, b) => b.count - a.count)}>
                    <XAxis dataKey="feedback" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#C4D600" />
                    </BarChart>
                </ResponsiveContainer>
                </div>
            </div>
            ) : (
                <p>Loading Feedback...</p>
            )}

        </div>
    );
};

export default FeedbackInfo;