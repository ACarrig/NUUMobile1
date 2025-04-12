import React from 'react';
import './Selector.css';

const SheetSelector = ({ sheets, selectedSheet, onSheetChange }) => {
  return (
    <div className="sheet-dropdown-container">
      <label htmlFor="sheet-dropdown">Selected Sheet: </label>
      <select id="sheet-dropdown" value={selectedSheet} onChange={onSheetChange}>
        <option value="">-- Choose a sheet --</option>
        {sheets.map((sheet, index) => (
          <option key={index} value={sheet}>{sheet}</option>
        ))}
      </select>
    </div>
  );
};

export default SheetSelector;
