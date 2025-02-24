import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'; // Import necessary components

import FileUpload from './components/FileUpload';  // Import your upload page component
import Navbar from './components/Navbar';  // Import your Navbar component
import Analysis from './DataBreakdown/Analysis'; // Import your Analysis component

const App = () => {
  return (
    <Router>
      <Navbar /> {/* Place Navbar here to make it appear on all pages */}

      <Routes>
        {/* Redirect default route (/) to /upload */}
        <Route path="/" element={<Navigate to="/upload" />} />

        {/* Upload Page Route */}
        <Route path="/upload" element={<FileUpload />} />

        {/* Analysis Page Route */}
        <Route path="/analysis" element={<Analysis />} />
        {/* You can add more routes here if needed, like About Page */}
      </Routes>
    </Router>
  );
}

export default App;
