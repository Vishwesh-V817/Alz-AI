import React, { useState, useEffect, useRef } from 'react';
import gsap from 'gsap';
import axios from 'axios';
import { Loader2, AlertCircle, CheckCircle } from 'lucide-react';

export default function DrugDiscovery() {
  const containerRef = useRef(null);
  const [smiles, setSmiles] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // Animate bars when results change
  useEffect(() => {
    if (results && containerRef.current) {
      const bars = containerRef.current.querySelectorAll('.sim-bar');

      // 1. Find the maximum similarity value in the current set (default to 1 to avoid division by zero)
      const maxVal = Math.max(...results.results.map(r => r.similarity), 0.1);

      gsap.fromTo(bars,
        { height: 0, opacity: 0 },
        {
          // 2. Scale the height: (Current Value / Max Value) * 90% 
          // This ensures the tallest bar always fills 90% of the container
          height: (i) => {
            const val = results.results[i]?.similarity || 0;
            return `${(val / maxVal) * 90}%`;
          },
          opacity: 1,
          duration: 1.2,
          stagger: 0.08,
          ease: "power3.out"
        }
      );
    }
  }, [results]);

  
  const analyzeDrug = async () => {
    if (!smiles.trim()) {
      setError('Please enter a SMILES string');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      // Call your FastAPI backend
      const response = await axios.get(`http://localhost:8000/repurpose`, {
        params: { query: smiles.trim() }
      });

      setResults(response.data);
    } catch (err) {
      console.error('Drug Discovery Error:', err);
      setError(err.response?.data?.error || 'Failed to analyze drug. Ensure backend is running on port 8001.');
    } finally {
      setLoading(false);
    }
  };

  // Handle Enter key
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      analyzeDrug();
    }
  };

  return (
    <div className="grid-2x2" ref={containerRef}>
      {/* Similarity Chart - Takes full left column - NOW SHOWS ALL 8 */}
      <div className="card" style={{ gridRow: 'span 2' }}>
        <h3>Protein Target Similarity (All 8 Targets)</h3>
        {results ? (
          <div className="bar-chart-container" style={{ height: '400px' }}>
            {results.results.map((target, i) => (
              <div key={i} className="bar-wrapper" style={{ width: '45px' }}>
                <span className="bar-value" style={{ fontSize: '10px', fontWeight: 800 }}>
                  {target.similarity}%
                </span>
                <div
                  className="sim-bar"
                  style={{
                    width: '28px',
                    background: target.similarity > 70 ? '#37c07e' :
                      target.similarity > 50 ? '#3b82f6' :
                        target.similarity > 30 ? '#f59e0b' : '#94a3b8'
                  }}
                ></div>
                <span className="bar-label" style={{ fontSize: '9px', fontWeight: 700 }}>
                  {target.target}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div style={{
            height: '400px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#94a3b8',
            border: '2px dashed #e2e8f0',
            borderRadius: '12px',
            fontSize: '14px',
            fontWeight: 600
          }}>
            {loading ? 'Analyzing molecular structure...' : 'Enter SMILES to see all 8 protein similarities'}
          </div>
        )}
      </div>

      {/* Input Card - Top Right */}
      <div className="card">
        <h3>Enter Drug SMILES</h3>
        <textarea
          rows="5"
          placeholder="Example: CC(C)NCC(COc1ccccc1CC=C)O (Donepezil)"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          onKeyPress={handleKeyPress}
          style={{
            border: error ? '2px solid #ef4444' : '1.5px solid #e2e8f0'
          }}
        ></textarea>

        {error && (
          <div style={{
            color: '#ef4444',
            fontSize: '12px',
            marginTop: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '5px'
          }}>
            <AlertCircle size={14} /> {error}
          </div>
        )}

        {results && (
          <div style={{
            color: '#37c07e',
            fontSize: '12px',
            marginTop: '10px',
            display: 'flex',
            alignItems: 'center',
            gap: '5px',
            fontWeight: 600
          }}>
            <CheckCircle size={14} /> Analysis Complete - 8 Targets Analyzed
          </div>
        )}

        <button
          className="btn-primary"
          style={{ marginTop: 15, opacity: loading ? 0.6 : 1 }}
          onClick={analyzeDrug}
          disabled={loading}
        >
          {loading ? (
            <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px' }}>
              <Loader2 size={16} className="spinning" /> Analyzing...
            </span>
          ) : (
            'Analyze Drug-Protein Interaction'
          )}
        </button>
        <p style={{ fontSize: '11px', color: '#94a3b8', marginTop: '10px', textAlign: 'center' }}>
          Tip: Press Ctrl+Enter to analyze
        </p>
      </div>

      {/* Binding Scores Card - Bottom Right */}
      <div className="card" style={{ maxHeight: '450px', overflowY: 'auto' }}>
        <h3>Binding Predictions</h3>
        {results ? (
          <div>
            {/* Show predictions for the top target */}
            {results.results[0] && (
              <div style={{ marginBottom: '15px', padding: '10px', background: '#f0fdf4', borderRadius: '8px', border: '1px solid #bbf7d0' }}>
                <p style={{ fontSize: '11px', fontWeight: 800, color: '#166534', margin: '0 0 8px 0' }}>
                  🏆 TOP TARGET: {results.results[0].target} ({results.results[0].uniprot})
                </p>
              </div>
            )}

            {results.results[0]?.predictions && Object.entries(results.results[0].predictions).map(([key, pred]) => (
              <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 15, marginBottom: 12 }}>
                <span style={{ width: 50, fontSize: 12, fontWeight: 900 }}>{key}</span>
                <div style={{ flex: 1, background: '#f1f5f9', height: 24, borderRadius: 10, overflow: 'hidden', position: 'relative' }}>
                  <div style={{
                    width: `${pred.confidence}%`,
                    background: pred.label === 'Elite' ? '#37c07e' :
                      pred.label === 'Strong' ? '#3b82f6' :
                        pred.label === 'Moderate' ? '#f59e0b' : '#ef4444',
                    height: '100%',
                    transition: 'width 0.5s ease'
                  }}></div>
                </div>
                <span style={{
                  fontSize: 11,
                  fontWeight: 900,
                  minWidth: '90px',
                  color: pred.label === 'Elite' ? '#37c07e' :
                    pred.label === 'Strong' ? '#3b82f6' :
                      pred.label === 'Moderate' ? '#f59e0b' : '#ef4444'
                }}>
                  {pred.label} ({pred.confidence}%)
                </span>
              </div>
            ))}

            {/* All Targets Summary - NOW SHOWS ALL 8 */}
            <div style={{ marginTop: '20px', paddingTop: '15px', borderTop: '1px solid #e2e8f0' }}>
              <p style={{ fontSize: '11px', fontWeight: 800, color: '#64748b', marginBottom: '10px' }}>
                ALL 8 TARGETS RANKED BY SIMILARITY:
              </p>
              {results.results.map((target, idx) => (
                <div key={idx} style={{
                  fontSize: '11px',
                  padding: '8px 10px',
                  background: idx === 0 ? '#f0fdf4' :
                    idx === 1 ? '#eff6ff' :
                      idx === 2 ? '#fef9c3' : '#f8fafc',
                  borderRadius: '6px',
                  marginBottom: '5px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  border: idx < 3 ? '1px solid #e2e8f0' : 'none'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{
                      fontWeight: 900,
                      color: idx === 0 ? '#166534' : idx === 1 ? '#1e40af' : idx === 2 ? '#854d0e' : '#64748b',
                      minWidth: '20px'
                    }}>
                      #{idx + 1}
                    </span>
                    <span style={{ fontWeight: 700 }}>{target.target}</span>
                    <span style={{ color: '#94a3b8', fontSize: '10px' }}>({target.uniprot})</span>
                  </div>
                  <span style={{
                    fontWeight: 800,
                    color: target.similarity > 70 ? '#37c07e' :
                      target.similarity > 50 ? '#3b82f6' :
                        target.similarity > 30 ? '#f59e0b' : '#94a3b8'
                  }}>
                    {target.similarity}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div style={{
            height: '200px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#94a3b8',
            border: '2px dashed #e2e8f0',
            borderRadius: '12px',
            fontSize: '13px'
          }}>
            Binding predictions will appear here
          </div>
        )}
      </div>
    </div>
  );
}