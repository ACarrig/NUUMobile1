import React from 'react';

function Navbar() {
  return (
    <div>
      <h1>Upload your files</h1>
      <nav className="navbar">
        <div className="logo">
          <img src="/assets/NuuMobileLogo.png" alt="Logo" />
        </div>
        <ul className="nav-links">
          <li><a href="#">About</a></li>
          <li><a href="#">Help</a></li>
        </ul>
      </nav>
    </div>
  );
}

export default Navbar;
