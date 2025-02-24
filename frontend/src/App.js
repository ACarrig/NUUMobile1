import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./Components/Navbar";
import FileUpload from "./Components/FileUpload";
import Footer from "./Components/Footer";
import Analysis from "./DataBreakdown/Analysis"; // Import Analysis component
import "./style.css"; // Importing the CSS file

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<FileUpload />} />
        <Route path="/analysis" element={<Analysis />} />
      </Routes>
      <Footer />
    </Router>
  );
}

export default App;
