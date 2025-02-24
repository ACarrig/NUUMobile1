import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Analysis from "./DataBreakdown/Analysis"; // Import Analysis component
import FileUpload from './components/FileUpload';  // Import your upload page component
import Navbar from './components/Navbar';  // Import your Navbar component

const App = () => {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<FileUpload />} />
        <Route path="/analysis" element={<Analysis />} />
      </Routes>
    </Router>
  );
}

export default App;
