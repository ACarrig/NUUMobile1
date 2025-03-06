import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";

import "./App.css";
import Navbar from "./components/Navbar";
import FileUpload from "./components/FileUpload";
import Analysis from "./DataBreakdown/Analysis";
import AppData from "./DataBreakdown/AppData";

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

        {/* App Usage Page Route */}
        <Route path="/appdata" element={<AppData />} />
        
      </Routes>
    </Router>
  );
}

export default App;
