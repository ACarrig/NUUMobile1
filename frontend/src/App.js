import React from 'react';
import Navbar from './Components/Navbar';
import FileUpload from './Components/FileUpload';
import Footer from './Components/Footer';
import './style.css';  // Importing the CSS file

function App() {
  return (
    <div>
      <Navbar />
      <FileUpload />
      <Footer />
    </div>
  );
}

export default App;
