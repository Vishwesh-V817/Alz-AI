import React, { useState, useEffect } from 'react';
import { UploadCloud, CheckCircle, AlertCircle, Activity, FileText } from 'lucide-react';

export default function AIAnalysis() {
  const [activeTab, setActiveTab] = useState('mri'); // 'mri' or 'clinical'

  // MRI Analysis State
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Clinical Analysis State
  const [clinicalLoading, setClinicalLoading] = useState(false);
  const [clinicalResult, setClinicalResult] = useState(null);
  const [clinicalError, setClinicalError] = useState(null);
  const [formData, setFormData] = useState({
    Age: 72,
    Gender: 1,
    Ethnicity: 0,
    EducationLevel: 2,
    BMI: 26.5,
    Smoking: 0,
    AlcoholConsumption: 3.5,
    PhysicalActivity: 4.2,
    DietQuality: 6.8,
    SleepQuality: 5.5,
    FamilyHistoryAlzheimers: 1,
    CardiovascularDisease: 0,
    Diabetes: 1,
    Depression: 0,
    HeadInjury: 0,
    Hypertension: 1,
    SystolicBP: 138.0,
    DiastolicBP: 85.0,
    CholesterolTotal: 210.0,
    CholesterolLDL: 135.0,
    CholesterolHDL: 48.0,
    CholesterolTriglycerides: 165.0,
    MMSE: 22.0,
    FunctionalAssessment: 4.5,
    ADL: 5.2,
    MemoryComplaints: 1,
    BehavioralProblems: 1,
    Confusion: 1,
    Disorientation: 0,
    PersonalityChanges: 1,
    DifficultyCompletingTasks: 1,
    Forgetfulness: 1
  });

  useEffect(() => {
    if (loading) {
      const loader = document.querySelector('.ring-loader');
      if (loader) {
        loader.style.animation = 'spin 2s linear infinite';
      }
    }
  }, [loading]);

  // MRI Analysis Functions
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    }
  };

  const runMRIAnalysis = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formDataObj = new FormData();
    formDataObj.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict-mri', {
        method: 'POST',
        body: formDataObj
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

  // Clinical Analysis Functions
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: ['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'Smoking',
        'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
        'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints',
        'BehavioralProblems', 'Confusion', 'Disorientation',
        'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'].includes(name)
        ? parseInt(value)
        : parseFloat(value)
    }));
  };

  const runClinicalAnalysis = async () => {
    setClinicalLoading(true);
    setClinicalError(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict-clinical', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        throw new Error('Clinical analysis failed');
      }

      const data = await response.json();
      setClinicalResult(data);
    } catch (err) {
      console.error("Clinical Analysis Error:", err);
      setClinicalError("Failed to analyze clinical data. Ensure the backend is running on port 8000.");
    } finally {
      setClinicalLoading(false);
    }
  };

  const loadExampleData = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/example-patient');
      const example = await response.json();
      setFormData(example);
    } catch (err) {
      setClinicalError("Could not load example data. Make sure backend is running.");
    }
  };

  const resetClinicalForm = () => {
    setFormData({
      Age: 72, Gender: 1, Ethnicity: 0, EducationLevel: 2, BMI: 26.5,
      Smoking: 0, AlcoholConsumption: 3.5, PhysicalActivity: 4.2,
      DietQuality: 6.8, SleepQuality: 5.5, FamilyHistoryAlzheimers: 1,
      CardiovascularDisease: 0, Diabetes: 1, Depression: 0,
      HeadInjury: 0, Hypertension: 1, SystolicBP: 138.0, DiastolicBP: 85.0,
      CholesterolTotal: 210.0, CholesterolLDL: 135.0, CholesterolHDL: 48.0,
      CholesterolTriglycerides: 165.0, MMSE: 22.0, FunctionalAssessment: 4.5,
      ADL: 5.2, MemoryComplaints: 1, BehavioralProblems: 1,
      Confusion: 1, Disorientation: 0, PersonalityChanges: 1,
      DifficultyCompletingTasks: 1, Forgetfulness: 1
    });
    setClinicalResult(null);
    setClinicalError(null);
  };

  return (
    <div>
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .tab-container {
          display: flex;
          gap: 10px;
          margin-bottom: 25px;
        }
        .tab-button {
          flex: 1;
          padding: 14px 20px;
          border: none;
          background: #f1f5f9;
          color: #64748b;
          border-radius: 10px;
          font-weight: 700;
          cursor: pointer;
          transition: all 0.3s;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          font-size: 14px;
        }
        .tab-button.active {
          background: var(--green);
          color: white;
          box-shadow: 0 4px 12px rgba(55, 192, 126, 0.3);
        }
        .tab-button:hover:not(.active) {
          background: #e2e8f0;
          color: #475569;
        }
        .form-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 15px;
          margin-bottom: 20px;
        }
        .form-group {
          display: flex;
          flex-direction: column;
        }
        .form-label {
          font-size: 11px;
          font-weight: 800;
          color: #64748b;
          text-transform: uppercase;
          margin-bottom: 6px;
        }
        .form-input, .form-select {
          padding: 10px 12px;
          border: 1.5px solid #e2e8f0;
          border-radius: 8px;
          background: #f8fafc;
          font-size: 13px;
          transition: border 0.3s;
          font-family: 'Inter', sans-serif;
        }
        .form-input:focus, .form-select:focus {
          outline: none;
          border-color: var(--green);
          background: white;
        }
        .section-header {
          font-size: 14px;
          font-weight: 800;
          color: #1e293b;
          margin: 25px 0 15px 0;
          padding-bottom: 8px;
          border-bottom: 2px solid var(--green);
          display: flex;
          align-items: center;
          gap: 8px;
        }
        .section-header:first-child {
          margin-top: 0;
        }
        .form-scroll {
          max-height: 500px;
          overflow-y: auto;
          padding-right: 10px;
        }
        .form-scroll::-webkit-scrollbar {
          width: 6px;
        }
        .form-scroll::-webkit-scrollbar-track {
          background: #f1f5f9;
          border-radius: 10px;
        }
        .form-scroll::-webkit-scrollbar-thumb {
          background: #cbd5e1;
          border-radius: 10px;
        }
        .form-scroll::-webkit-scrollbar-thumb:hover {
          background: #94a3b8;
        }
        .risk-badge {
          display: inline-block;
          padding: 8px 16px;
          border-radius: 20px;
          font-weight: 800;
          font-size: 13px;
          margin-top: 10px;
        }
        .risk-high { background: #fee; color: #991b1b; }
        .risk-moderate { background: #fef3c7; color: #92400e; }
        .risk-low { background: #dbeafe; color: #1e40af; }
      `}</style>

      {/* Tab Navigation */}
      <div className="tab-container">
        <button
          className={`tab-button ${activeTab === 'mri' ? 'active' : ''}`}
          onClick={() => setActiveTab('mri')}
        >
          <Activity size={18} />
          MRI Scan Analysis
        </button>
        <button
          className={`tab-button ${activeTab === 'clinical' ? 'active' : ''}`}
          onClick={() => setActiveTab('clinical')}
        >
          <FileText size={18} />
          Clinical Data Assessment
        </button>
      </div>

      {/* MRI Analysis Tab */}
      {activeTab === 'mri' && (
        <div className="mid-grid" style={{ gridTemplateColumns: '1fr 1fr' }}>
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
                  <button className="btn-green" style={{ background: '#1a1f2b' }} onClick={runMRIAnalysis}>
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

                <div style={{ marginTop: '20px' }}>
                  <p style={{ fontSize: '11px', fontWeight: 'bold', color: '#64748b', marginBottom: '12px' }}>
                    CONFIDENCE SCORES BY CLASS:
                  </p>

                  {result.all_probs && Object.entries(result.all_probs)
                    .sort(([, a], [, b]) => b - a)
                    .map(([className, prob]) => {
                      const percentage = (prob * 100).toFixed(2);
                      const isTopClass = className === result.prediction;
                      let barColor = '#cbd5e1';
                      if (isTopClass) {
                        barColor = '#37c07e';
                      } else if (prob > 0.3) {
                        barColor = '#f59e0b';
                      }

                      return (
                        <div key={className} style={{ marginBottom: '15px' }}>
                          <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            marginBottom: '8px'
                          }}>
                            <span style={{
                              fontSize: '13px',
                              fontWeight: isTopClass ? 800 : 600,
                              color: isTopClass ? '#1e293b' : '#64748b'
                            }}>
                              {className === '0_Normal' ? '0 NORMAL' :
                                className === '1_Mild_Impairment' ? '1 MILD IMPAIRMENT' :
                                  className === '2_Moderate_Impairment' ? '2 MODERATE IMPAIRMENT' :
                                    className.replace(/_/g, ' ').toUpperCase()}
                            </span>
                            <span style={{
                              fontSize: '13px',
                              fontWeight: 800,
                              color: isTopClass ? '#37c07e' : '#94a3b8'
                            }}>
                              {percentage}%
                            </span>
                          </div>
                          <div style={{
                            width: '100%',
                            height: '12px',
                            background: '#f1f5f9',
                            borderRadius: '6px',
                            overflow: 'hidden'
                          }}>
                            <div style={{
                              width: `${percentage}%`,
                              height: '100%',
                              background: barColor,
                              transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
                              borderRadius: '6px'
                            }}></div>
                          </div>
                        </div>
                      );
                    })}
                </div>

                <div style={{
                  marginTop: '20px',
                  padding: '15px',
                  background: '#f8fafc',
                  borderRadius: '10px',
                  border: '1px solid #e2e8f0'
                }}>
                  <p style={{
                    fontSize: '10px',
                    fontWeight: 'bold',
                    color: '#64748b',
                    marginBottom: '10px',
                    textTransform: 'uppercase'
                  }}>
                    Classification Details
                  </p>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px' }}>
                    {result.all_probs && Object.entries(result.all_probs).map(([className, prob]) => {
                      const percentage = (prob * 100).toFixed(1);
                      const isTopClass = className === result.prediction;

                      return (
                        <div key={className} style={{
                          padding: '10px',
                          background: isTopClass ? '#ecfdf5' : 'white',
                          border: isTopClass ? '2px solid #37c07e' : '1px solid #e2e8f0',
                          borderRadius: '8px',
                          textAlign: 'center'
                        }}>
                          <div style={{
                            fontSize: '20px',
                            fontWeight: 900,
                            color: isTopClass ? '#37c07e' : '#94a3b8',
                            marginBottom: '5px'
                          }}>
                            {percentage}%
                          </div>
                          <div style={{
                            fontSize: '9px',
                            fontWeight: 700,
                            color: '#64748b',
                            textTransform: 'uppercase'
                          }}>
                            {className.replace(/_/g, ' ')}
                          </div>
                          {isTopClass && (
                            <div style={{
                              marginTop: '5px',
                              fontSize: '8px',
                              color: '#37c07e',
                              fontWeight: 800
                            }}>
                              ✓ PREDICTED
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </>
            ) : (
              <div style={{
                height: '400px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#94a3b8',
                border: '1px dashed #e2e8f0',
                borderRadius: '12px',
                fontSize: '14px',
                fontWeight: 600
              }}>
                {loading ? "AI is thinking..." : "Awaiting patient data..."}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Clinical Assessment Tab */}
      {activeTab === 'clinical' && (
        <div className="mid-grid" style={{ gridTemplateColumns: '1.2fr 0.8fr' }}>
          <div className="card">
            <h3>Clinical Data Assessment</h3>

            <div className="form-scroll">
              {/* Demographics */}
              <div className="section-header">📋 Demographics</div>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Age</label>
                  <input className="form-input" type="number" name="Age" value={formData.Age} onChange={handleInputChange} min="40" max="100" />
                </div>
                <div className="form-group">
                  <label className="form-label">Gender</label>
                  <select className="form-select" name="Gender" value={formData.Gender} onChange={handleInputChange}>
                    <option value="0">Male</option>
                    <option value="1">Female</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Ethnicity</label>
                  <select className="form-select" name="Ethnicity" value={formData.Ethnicity} onChange={handleInputChange}>
                    <option value="0">Caucasian</option>
                    <option value="1">African American</option>
                    <option value="2">Asian</option>
                    <option value="3">Other</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Education Level</label>
                  <select className="form-select" name="EducationLevel" value={formData.EducationLevel} onChange={handleInputChange}>
                    <option value="0">None</option>
                    <option value="1">High School</option>
                    <option value="2">Bachelor's</option>
                    <option value="3">Higher</option>
                  </select>
                </div>
              </div>

              {/* Physical Health */}
              <div className="section-header">💪 Physical Health</div>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">BMI</label>
                  <input className="form-input" type="number" name="BMI" value={formData.BMI} onChange={handleInputChange} step="0.1" min="15" max="50" />
                </div>
                <div className="form-group">
                  <label className="form-label">Smoking</label>
                  <select className="form-select" name="Smoking" value={formData.Smoking} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Alcohol (units/week)</label>
                  <input className="form-input" type="number" name="AlcoholConsumption" value={formData.AlcoholConsumption} onChange={handleInputChange} step="0.1" min="0" max="20" />
                </div>
                <div className="form-group">
                  <label className="form-label">Physical Activity (0-10)</label>
                  <input className="form-input" type="number" name="PhysicalActivity" value={formData.PhysicalActivity} onChange={handleInputChange} step="0.1" min="0" max="10" />
                </div>
                <div className="form-group">
                  <label className="form-label">Diet Quality (0-10)</label>
                  <input className="form-input" type="number" name="DietQuality" value={formData.DietQuality} onChange={handleInputChange} step="0.1" min="0" max="10" />
                </div>
                <div className="form-group">
                  <label className="form-label">Sleep Quality (0-10)</label>
                  <input className="form-input" type="number" name="SleepQuality" value={formData.SleepQuality} onChange={handleInputChange} step="0.1" min="0" max="10" />
                </div>
              </div>

              {/* Medical History */}
              <div className="section-header">🏥 Medical History</div>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Family History Alzheimer's</label>
                  <select className="form-select" name="FamilyHistoryAlzheimers" value={formData.FamilyHistoryAlzheimers} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Cardiovascular Disease</label>
                  <select className="form-select" name="CardiovascularDisease" value={formData.CardiovascularDisease} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Diabetes</label>
                  <select className="form-select" name="Diabetes" value={formData.Diabetes} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Depression</label>
                  <select className="form-select" name="Depression" value={formData.Depression} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Head Injury</label>
                  <select className="form-select" name="HeadInjury" value={formData.HeadInjury} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Hypertension</label>
                  <select className="form-select" name="Hypertension" value={formData.Hypertension} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
              </div>

              {/* Clinical Measurements */}
              <div className="section-header">🩺 Clinical Measurements</div>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Systolic BP (mmHg)</label>
                  <input className="form-input" type="number" name="SystolicBP" value={formData.SystolicBP} onChange={handleInputChange} step="0.1" min="80" max="200" />
                </div>
                <div className="form-group">
                  <label className="form-label">Diastolic BP (mmHg)</label>
                  <input className="form-input" type="number" name="DiastolicBP" value={formData.DiastolicBP} onChange={handleInputChange} step="0.1" min="40" max="130" />
                </div>
                <div className="form-group">
                  <label className="form-label">Total Cholesterol</label>
                  <input className="form-input" type="number" name="CholesterolTotal" value={formData.CholesterolTotal} onChange={handleInputChange} step="0.1" min="100" max="400" />
                </div>
                <div className="form-group">
                  <label className="form-label">LDL Cholesterol</label>
                  <input className="form-input" type="number" name="CholesterolLDL" value={formData.CholesterolLDL} onChange={handleInputChange} step="0.1" min="50" max="250" />
                </div>
                <div className="form-group">
                  <label className="form-label">HDL Cholesterol</label>
                  <input className="form-input" type="number" name="CholesterolHDL" value={formData.CholesterolHDL} onChange={handleInputChange} step="0.1" min="20" max="100" />
                </div>
                <div className="form-group">
                  <label className="form-label">Triglycerides</label>
                  <input className="form-input" type="number" name="CholesterolTriglycerides" value={formData.CholesterolTriglycerides} onChange={handleInputChange} step="0.1" min="50" max="500" />
                </div>
              </div>

              {/* Cognitive Assessments */}
              <div className="section-header">🧩 Cognitive Assessments</div>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">MMSE Score (0-30)</label>
                  <input className="form-input" type="number" name="MMSE" value={formData.MMSE} onChange={handleInputChange} step="0.1" min="0" max="30" />
                </div>
                <div className="form-group">
                  <label className="form-label">Functional Assessment</label>
                  <input className="form-input" type="number" name="FunctionalAssessment" value={formData.FunctionalAssessment} onChange={handleInputChange} step="0.1" min="0" max="10" />
                </div>
                <div className="form-group">
                  <label className="form-label">ADL Score</label>
                  <input className="form-input" type="number" name="ADL" value={formData.ADL} onChange={handleInputChange} step="0.1" min="0" max="10" />
                </div>
              </div>

              {/* Clinical Symptoms */}
              <div className="section-header">⚠️ Clinical Symptoms</div>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Memory Complaints</label>
                  <select className="form-select" name="MemoryComplaints" value={formData.MemoryComplaints} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Behavioral Problems</label>
                  <select className="form-select" name="BehavioralProblems" value={formData.BehavioralProblems} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Confusion</label>
                  <select className="form-select" name="Confusion" value={formData.Confusion} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Disorientation</label>
                  <select className="form-select" name="Disorientation" value={formData.Disorientation} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Personality Changes</label>
                  <select className="form-select" name="PersonalityChanges" value={formData.PersonalityChanges} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Difficulty Completing Tasks</label>
                  <select className="form-select" name="DifficultyCompletingTasks" value={formData.DifficultyCompletingTasks} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Forgetfulness</label>
                  <select className="form-select" name="Forgetfulness" value={formData.Forgetfulness} onChange={handleInputChange}>
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              <button className="btn-primary" onClick={loadExampleData} style={{ flex: 1, background: '#3b82f6' }}>
                Load Example
              </button>
              <button className="btn-primary" onClick={resetClinicalForm} style={{ flex: 1, background: '#6c757d' }}>
                Reset Form
              </button>
              <button className="btn-primary" onClick={runClinicalAnalysis} disabled={clinicalLoading} style={{ flex: 2 }}>
                {clinicalLoading ? 'Analyzing...' : 'Analyze Patient'}
              </button>
            </div>

            {clinicalError && (
              <div style={{ color: '#ef4444', fontSize: '12px', marginTop: '15px', textAlign: 'center' }}>
                <AlertCircle size={14} style={{ verticalAlign: 'middle', marginRight: '5px' }} /> {clinicalError}
              </div>
            )}
          </div>

          {/* Clinical Results Panel */}
          <div className="card">
            <h3>Clinical Assessment Results</h3>

            {clinicalResult ? (
              <div>
                <p style={{ fontSize: 11, color: '#94a3b8', margin: '0 0 10px 0', fontWeight: 700 }}>
                  Analysis Complete <CheckCircle size={10} color="#37c07e" style={{ verticalAlign: 'middle' }} />
                </p>

                {/* Diagnosis Badge */}
                <div style={{
                  padding: '15px',
                  background: clinicalResult.diagnosis === 'Positive' ? '#fef3c7' : '#d4edda',
                  border: `2px solid ${clinicalResult.diagnosis === 'Positive' ? '#fbbf24' : '#10b981'}`,
                  borderRadius: '12px',
                  textAlign: 'center',
                  marginBottom: '20px'
                }}>
                  <div style={{
                    fontSize: '20px',
                    fontWeight: 900,
                    color: clinicalResult.diagnosis === 'Positive' ? '#92400e' : '#065f46',
                    marginBottom: '5px'
                  }}>
                    {clinicalResult.diagnosis}
                  </div>
                  <div style={{
                    fontSize: '11px',
                    color: clinicalResult.diagnosis === 'Positive' ? '#78350f' : '#064e3b',
                    fontWeight: 600
                  }}>
                    Alzheimer's Diagnosis
                  </div>
                </div>

                {/* Confidence Score */}
                <div style={{ marginBottom: '20px' }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    marginBottom: '8px'
                  }}>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: '#64748b' }}>
                      Confidence Level
                    </span>
                    <span style={{ fontSize: '12px', fontWeight: 800, color: '#1e293b' }}>
                      {clinicalResult.confidence}%
                    </span>
                  </div>
                  <div style={{
                    width: '100%',
                    height: '20px',
                    background: '#f1f5f9',
                    borderRadius: '10px',
                    overflow: 'hidden'
                  }}>
                    <div style={{
                      width: `${clinicalResult.confidence}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #667eea 0%, #764ba2 100%)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: 'white',
                      fontSize: '10px',
                      fontWeight: 800,
                      transition: 'width 0.8s ease'
                    }}>
                      {clinicalResult.confidence >= 20 && `${clinicalResult.confidence}%`}
                    </div>
                  </div>
                </div>

                {/* Risk Level */}
                <div style={{
                  padding: '15px',
                  background: '#f8fafc',
                  borderRadius: '10px',
                  border: '1px solid #e2e8f0',
                  marginBottom: '20px'
                }}>
                  <div style={{ fontSize: '11px', fontWeight: 800, color: '#64748b', marginBottom: '8px' }}>
                    RISK ASSESSMENT
                  </div>
                  <span className={`risk-badge ${clinicalResult.risk_level === 'High' ? 'risk-high' :
                      clinicalResult.risk_level === 'Moderate' ? 'risk-moderate' : 'risk-low'
                    }`}>
                    {clinicalResult.risk_level} Risk
                  </span>
                </div>

                {/* Clinical Summary */}
                <div style={{
                  padding: '15px',
                  background: '#f8fafc',
                  borderRadius: '10px',
                  border: '1px solid #e2e8f0'
                }}>
                  <div style={{ fontSize: '11px', fontWeight: 800, color: '#64748b', marginBottom: '12px' }}>
                    CLINICAL SUMMARY
                  </div>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    padding: '10px 0',
                    borderBottom: '1px solid #e2e8f0',
                    fontSize: '12px'
                  }}>
                    <span style={{ fontWeight: 600, color: '#64748b' }}>Prediction Code:</span>
                    <span style={{ fontWeight: 800, color: '#1e293b' }}>{clinicalResult.prediction}</span>
                  </div>
                  <div style={{
                    padding: '10px 0',
                    fontSize: '12px'
                  }}>
                    <span style={{ fontWeight: 600, color: '#64748b' }}>Status:</span>
                    <div style={{ marginTop: '5px', fontWeight: 600, color: '#475569', fontSize: '11px', lineHeight: '1.6' }}>
                      {clinicalResult.message}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{
                height: '450px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#94a3b8',
                border: '2px dashed #e2e8f0',
                borderRadius: '12px',
                fontSize: '13px',
                padding: '20px',
                textAlign: 'center'
              }}>
                {clinicalLoading ? (
                  <>
                    <div className="ring-loader" style={{ marginBottom: '15px' }}></div>
                    <p style={{ fontWeight: 700 }}>Analyzing clinical data...</p>
                  </>
                ) : (
                  <>
                    <FileText size={40} style={{ marginBottom: '15px', opacity: 0.5 }} />
                    <p style={{ fontWeight: 600 }}>Fill out the clinical assessment form and click "Analyze Patient" to get diagnosis results.</p>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}