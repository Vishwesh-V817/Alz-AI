import React, { useState, useEffect } from 'react';
import { UploadCloud, CheckCircle, AlertCircle } from 'lucide-react';

export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (loading) {
      const loader = document.querySelector('.ring-loader');
      if (loader) {
        loader.style.animation = 'spin 2s linear infinite';
      }
    }
  }, [loading]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    }
  };

  const runAnalysis = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict-mri', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Analysis Error:", err);
      setError("Failed to analyze file. Ensure the backend is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      <section className="stats-grid">
        <div className="card stat-card">
          <label>Total Patients:</label>
          <h1>245 <small>1265</small></h1>
          <div style={{ display: 'flex', gap: 4 }}>
            {[1, 2, 3, 4, 5].map(i => (
              <div key={i} style={{ width: 5, height: 5, background: i === 4 ? '#37c07e' : '#cbd5e1', borderRadius: '50%' }}></div>
            ))}
          </div>
        </div>
        <div className="card stat-card">
          <label>Treatment Completed</label>
          <h1>82 <small>1623</small></h1>
          <label style={{ color: '#94a3b8', fontSize: '10px' }}>Undergoing Treatment</label>
        </div>
        <div className="card stat-card">
          <label>Treatment Completed</label>
          <h1>163 <small>1633</small></h1>
          <label style={{ color: '#94a3b8', fontSize: '10px' }}>Undergoing Treatment</label>
        </div>
        <div className="card stat-card">
          <label>Patient Activity Trend</label>
          <div style={{ height: 45, background: '#eef2f6', marginTop: 10, borderRadius: 8, display: 'flex', alignItems: 'flex-end', padding: '0 5px', gap: 3 }}>
            {[20, 45, 30, 60, 40, 75, 90].map((h, i) => (
              <div key={i} style={{ flex: 1, height: h + '%', background: '#37c07e', borderRadius: 2 }}></div>
            ))}
          </div>
        </div>
      </section>

      <div className="mid-grid">
        <div className="card">
          <h3>AI Analysis Module</h3>
          <div className="upload-dashed" style={{ borderColor: file ? '#37c07e' : '#cbd5e1' }}>
            <UploadCloud size={35} color={file ? "#37c07e" : "#94a3b8"} />
            <p style={{ fontSize: 13, color: '#64748b', margin: '15px 0', fontWeight: 600 }}>
              {file ? file.name : "Upload Patient Data (MRI ZIP, DICOM, or Image)"}
            </p>
            <input
              type="file"
              id="mri-upload"
              hidden
              onChange={handleFileChange}
              accept=".zip, .dcm, image/*"
            />
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
              <button className="btn-green" onClick={() => document.getElementById('mri-upload').click()}>
                Browse Files
              </button>
              {file && !loading && (
                <button className="btn-green" style={{ background: '#1a1f2b' }} onClick={runAnalysis}>
                  Analyze Now
                </button>
              )}
            </div>
          </div>

          {loading && (
            <div className="proc-ui">
              <div className="ring-loader"></div>
              <p style={{ fontSize: 12, color: '#64748b', fontWeight: 700 }}>Processing AI Results...</p>
            </div>
          )}

          {error && (
            <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '10px', textAlign: 'center' }}>
              <AlertCircle size={14} style={{ verticalAlign: 'middle', marginRight: '5px' }} /> {error}
            </div>
          )}
        </div>

        <div className="card">
          <h3>Diagnosis Report</h3>
          {result ? (
            <>
              <p style={{ fontSize: 11, color: '#94a3b8', margin: 0, fontWeight: 700 }}>
                Analysis Complete <CheckCircle size={10} color="#37c07e" />
              </p>
              <h2 style={{ margin: '8px 0', fontSize: 22, fontWeight: 800 }}>
                {result.prediction.replace(/_/g, ' ').toUpperCase()}
              </h2>
              <p style={{ fontSize: 13, color: '#37c07e', fontWeight: 800 }}>
                Top Confidence: {(result.confidence * 100).toFixed(2)}%
              </p>

              {/* Confidence Scores for All 3 Classes */}
              <div style={{ marginTop: '20px' }}>
                <p style={{ fontSize: '11px', fontWeight: 'bold', color: '#64748b', marginBottom: '12px' }}>
                  CONFIDENCE SCORES BY CLASS:
                </p>

                {result.all_probs && Object.entries(result.all_probs).map(([className, prob]) => {
                  const percentage = (prob * 100).toFixed(2);
                  const isTopClass = className === result.prediction;

                  return (
                    <div key={className} style={{ marginBottom: '12px' }}>
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '6px'
                      }}>
                        <span style={{
                          fontSize: '12px',
                          fontWeight: isTopClass ? 800 : 600,
                          color: isTopClass ? '#1e293b' : '#64748b'
                        }}>
                          {className.replace(/_/g, ' ').toUpperCase()}
                        </span>
                        <span style={{
                          fontSize: '12px',
                          fontWeight: 800,
                          color: isTopClass ? '#37c07e' : '#94a3b8'
                        }}>
                          {percentage}%
                        </span>
                      </div>
                      <div style={{
                        width: '100%',
                        height: '8px',
                        background: '#f1f5f9',
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          width: `${percentage}%`,
                          height: '100%',
                          background: isTopClass ? '#37c07e' : '#cbd5e1',
                          transition: 'width 0.5s ease',
                          borderRadius: '4px'
                        }}></div>
                      </div>
                    </div>
                  );
                })}
              </div>


            </>
          ) : (
            <div style={{
              height: '300px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#94a3b8',
              border: '1px dashed #e2e8f0',
              borderRadius: '12px'
            }}>
              {loading ? "AI is thinking..." : "Awaiting patient data..."}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h3>Drug Discovery & Protein Analysis</h3>
        <div className="protein-box">
          <div className="protein-viz">🧬</div>
          <div className="protein-details">
            <div>
              <h4 style={{ fontSize: 14, marginBottom: 10 }}>Suggested Medications:</h4>
              <ul style={{ fontSize: 13, color: '#475569', lineHeight: '1.8' }}>
                <li>Donepezil</li>
                <li>Galantamine</li>
                <li>Memantine</li>
              </ul>
            </div>
            <div>
              <h4 style={{ fontSize: 14, marginBottom: 10 }}>Research Feed</h4>
              <p style={{ fontSize: 13, color: '#64748b', lineHeight: '1.6' }}>• Clinical Trial Update: New Tau-Protein Inhibitor</p>
              <p style={{ fontSize: 13, color: '#64748b', lineHeight: '1.6' }}>• Bio-marker discovery in CSF fluid</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}