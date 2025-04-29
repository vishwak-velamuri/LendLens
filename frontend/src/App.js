import React, { useState } from 'react';
import './App.css';
import UploadForm from './components/UploadForm';
import ResultsDashboard from './components/ResultsDashboard';

function App() {
  // State variables
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Handler function for receiving analysis results
  const handleAnalyze = (data, err = null) => {
    if (err) {
      console.error("API Error:", err);
      setError("Something went wrong: " + err.message);
      setResult(null);
      return;
    }
    
    if (data) {
      console.log("Received data:", data);
      setResult(data);
      setError('');
    } else {
      setError("Received empty data from server");
      setResult(null);
    }
  };

  // Handler function to reset the form
  const handleReset = () => {
    setResult(null);
    setError('');
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Loan Application Analyzer</h1>
      
      {error && (
        <div className="error-message">
          <div className="badge badge-danger">{error}</div>
          <button 
            className="btn btn-sm" 
            onClick={handleReset}
            style={{ marginLeft: '10px' }}
          >
            Try Again
          </button>
        </div>
      )}
      
      {loading ? (
        <div className="loading-spinner">
          <p>Processing your request...</p>
        </div>
      ) : result ? (
        <ResultsDashboard 
          data={result} 
          onReset={handleReset} 
        />
      ) : (
        <UploadForm 
          onAnalyze={handleAnalyze} 
          isLoading={loading} 
          setIsLoading={setLoading}
        />
      )}
    </div>
  );
}

export default App;