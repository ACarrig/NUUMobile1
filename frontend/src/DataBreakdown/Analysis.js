import React, { useEffect, useState } from "react";

function Analysis() {
  const [data, setData] = useState(null);

  useEffect(() => {
    // Fetch the analysis results from backend
    fetch("http://127.0.0.1:5001/app_usage")
      .then((response) => response.json())
      .then((data) => setData(data))
      .catch((error) => console.error("Error fetching analysis:", error));
  }, []);

  return (
    <div>
      <h2>Analysis Results</h2>
      {data ? (
        <pre>{JSON.stringify(data, null, 2)}</pre>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
}

export default Analysis;
