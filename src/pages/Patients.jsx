import React from 'react';
export default function Patients() {
  const data = [
    {id: '0001', group: 'Normal', age: 87, mmse: 70, cdr: 0},
    {id: '0002', group: 'Moderate Impairment', age: 75, mmse: 23, cdr: 0.5},
    {id: '0004', group: 'Mild Impairment', age: 88, mmse: 30, cdr: 0},
    {id: '0005', group: 'Mild Impariment', age: 80, mmse: 40, cdr: 0},
  ];
  return (
    <div className="card">
      <h3>Patient Data</h3>
      <table className="table-ui">
        <thead><tr><th>Subject ID</th><th>Group</th><th>Age</th><th>MMSE</th><th>CDR Score</th></tr></thead>
        <tbody>
          {data.map(p => (
            <tr key={p.id}><td><b>{p.id}</b></td><td>{p.group}</td><td>{p.age}</td><td>{p.mmse}</td><td>{p.cdr}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}