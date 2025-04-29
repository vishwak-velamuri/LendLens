import React, { useState } from 'react';
import { analyzeStatement } from '../services/api';

function UploadForm({ onAnalyze, isLoading, setIsLoading }) {
  // Local state
  const [file, setFile] = useState(null);
  const [loanAmount, setLoanAmount] = useState('');
  const [downPayment, setDownPayment] = useState('');
  const [interestRate, setInterestRate] = useState('');
  const [termMonths, setTermMonths] = useState('');
  const [validationError, setValidationError] = useState('');
  
  // Reset form fields
  const resetForm = () => {
    setFile(null);
    setLoanAmount('');
    setDownPayment('');
    setInterestRate('');
    setTermMonths('');
    setValidationError('');
    
    // Reset the file input by clearing its value
    const fileInput = document.getElementById('file-upload');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Form validation
    if (!file || !loanAmount || !downPayment || !interestRate || !termMonths) {
      setValidationError('All fields are required');
      return;
    }
    
    // Clear previous validation errors
    setValidationError('');
    
    // Build FormData object
    const formData = new FormData();
    formData.append('file', file);
    formData.append('loan_amount', Number(loanAmount));
    formData.append('down_payment', Number(downPayment));
    formData.append('interest_rate', Number(interestRate));
    formData.append('term_months', Number(termMonths));
    
    try {
      // Set loading state
      setIsLoading(true);
      
      // Call the API service
      const result = await analyzeStatement(formData);
      
      // Pass result to parent component
      onAnalyze(result);
    } catch (err) {
      // Pass error to parent component
      onAnalyze(null, err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form className="upload-form" onSubmit={handleSubmit}>
      <h2 className="form-title">LendLens: Bank Statement Analyzer</h2>
      
      {validationError && (
        <div className="badge badge-danger">{validationError}</div>
      )}
      
      <div className="form-field">
        <label htmlFor="file-upload">Bank Statement (PDF)</label>
        <div className="file-upload-container">
          <input
            id="file-upload"
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />
          {file && <p className="file-name">{file.name}</p>}
        </div>
      </div>
      
      <div className="form-field">
        <label htmlFor="loan-amount">Loan Amount ($)</label>
        <input
          id="loan-amount"
          type="number"
          min="0"
          step="1000"
          value={loanAmount}
          onChange={(e) => setLoanAmount(e.target.value)}
          placeholder="e.g. 250000"
        />
      </div>
      
      <div className="form-field">
        <label htmlFor="down-payment">Down Payment ($)</label>
        <input
          id="down-payment"
          type="number"
          min="0"
          step="1000"
          value={downPayment}
          onChange={(e) => setDownPayment(e.target.value)}
          placeholder="e.g. 50000"
        />
      </div>
      
      <div className="form-field">
        <label htmlFor="interest-rate">Interest Rate (%)</label>
        <input
          id="interest-rate"
          type="number"
          min="0"
          step="0.1"
          value={interestRate}
          onChange={(e) => setInterestRate(e.target.value)}
          placeholder="e.g. 4.5"
        />
      </div>
      
      <div className="form-field">
        <label htmlFor="term-months">Loan Term (Months)</label>
        <input
          id="term-months"
          type="number"
          min="12"
          step="12"
          value={termMonths}
          onChange={(e) => setTermMonths(e.target.value)}
          placeholder="e.g. 360"
        />
      </div>
      
      <div className="button-group">
        <button 
          type="submit" 
          className="btn btn-primary"
          disabled={isLoading}
        >
          {isLoading ? 'Processing...' : 'Analyze Statement'}
        </button>
        
        <button 
          type="button" 
          className="btn btn-outline"
          onClick={resetForm}
          disabled={isLoading}
        >
          Reset Form
        </button>
      </div>
    </form>
  );
}

export default UploadForm;