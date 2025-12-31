import React from 'react';
import { Clock, MapPin } from 'lucide-react';
export default function Appointments() {
  return (
    <div className="card">
      <h3>Medical Schedule</h3>
      <div className="card" style={{background:'#f1f5f9', display:'flex', justifyContent:'space-between', alignItems:'center'}}>
        <div><h4 style={{margin:0}}>S-0002 - MRI Session</h4><p style={{fontSize:12, margin:0, color:'#64748b'}}><MapPin size={12}/> Radiology Suite 4</p></div>
        <div style={{color:'#37c07e', fontWeight:900}}><Clock size={16}/> 10:30 AM</div>
      </div>
    </div>
  );
}