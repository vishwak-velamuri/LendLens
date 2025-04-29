import React from 'react';

function ResultsDashboard({ data, onReset }) {
  // Defensive check
  if (!data) {
    return (
      <div className="error-container">
        <h2>Error Loading Results</h2>
        <p>Unable to process results. Please try again.</p>
        <button 
          className="btn btn-outline" 
          onClick={onReset}
        >
          Back to Form
        </button>
      </div>
    );
  }

  console.log("Dashboard received data:", data);

  // Safely extract data with null checks
  const decision = data.decision || 'UNKNOWN';
  const confidence = data.confidence !== undefined ? data.confidence : 0;
  
  // Extract metrics data
  const metrics = data.metrics || {};
  const monthlyTotals = metrics.monthly_totals || {};
  const categoryTotals = metrics.category_totals || {
    deposit: 0,
    withdrawal: 0,
    withdrawal_regular_bill: 0
  };

  // Format currency values
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    }).format(value || 0);
  };

  // Check if we have monthly data to display
  const hasMonthlyData = Object.keys(monthlyTotals).length > 0;

  return (
    <div className="results-dashboard">
      {/* Decision Banner */}
      <div className={`decision-banner decision-${decision}`}>
        {decision} {confidence !== undefined && `(${(confidence * 100).toFixed(1)}%)`}
      </div>

      <h2 className="dashboard-title">Financial Summary</h2>

      {/* Metrics Grid */}
      <div className="metrics-grid">
        <div className="metric-card">
          <h4>Total Deposits</h4>
          <p>{formatCurrency(categoryTotals.deposit)}</p>
        </div>
        <div className="metric-card">
          <h4>Total Withdrawals</h4>
          <p>{formatCurrency(categoryTotals.withdrawal)}</p>
        </div>
        <div className="metric-card">
          <h4>Regular Bills</h4>
          <p>{formatCurrency(categoryTotals.withdrawal_regular_bill)}</p>
        </div>
      </div>

      {hasMonthlyData && (
        <>
          <h3 className="dashboard-subtitle">Monthly Cash Flow</h3>

          {/* Monthly Flows Table */}
          <table>
            <thead>
              <tr>
                <th>Month</th>
                <th>Deposits</th>
                <th>Withdrawals</th>
                <th>Net Flow</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(monthlyTotals).map(month => {
                const monthData = monthlyTotals[month] || {};
                const deposits = monthData.deposits || 0;
                const withdrawals = monthData.withdrawals || 0;
                
                return (
                  <tr key={month}>
                    <td>{month}</td>
                    <td>{formatCurrency(deposits)}</td>
                    <td>{formatCurrency(withdrawals)}</td>
                    <td>{formatCurrency(deposits - withdrawals)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      {/* Summary Section */}
      <div className="card">
        <h3 className="dashboard-subtitle">Analysis Summary</h3>
        <p>
          Based on the financial data provided, the loan application has been 
          <strong className={decision === 'APPROVE' ? 'badge-success' : 'badge-danger'}>
            {` ${decision.toLowerCase()}`}
          </strong>
          {confidence !== undefined && ` with ${(confidence * 100).toFixed(1)}% confidence`}.
        </p>
      </div>
      
      <button 
        className="btn btn-outline" 
        onClick={onReset}
        style={{ marginTop: '2rem' }}
      >
        Analyze Another Statement
      </button>
    </div>
  );
}

export default ResultsDashboard;