import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, Stethoscope, BrainCircuit, Pill, Bell, User } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Patients from './pages/Patients';
import Appointments from './pages/Appointments';
import AIAnalysis from './pages/AIAnalysis';
import DrugDiscovery from './pages/DrugDiscovery';
import './App.css';

export default function App() {
  return (
    <Router>
      <div className="layout">
        <aside className="sidebar">
          <span className="logo-text">ALZ - AI</span>
          <nav>
            <NavLink to="/" className={({isActive})=>isActive?'nav-item active':'nav-item'}><LayoutDashboard size={20}/> Analytics</NavLink>
            <NavLink to="/appointments" className={({isActive})=>isActive?'nav-item active':'nav-item'}><Stethoscope size={20}/> Appointments</NavLink>
            <NavLink to="/patients" className={({isActive})=>isActive?'nav-item active':'nav-item'}><Users size={20}/> Patients</NavLink>
            <NavLink to="/ai-analysis" className={({isActive})=>isActive?'nav-item active':'nav-item'}><BrainCircuit size={20}/> AI Analysis</NavLink>
            <NavLink to="/drug-discovery" className={({isActive})=>isActive?'nav-item active':'nav-item'}><Pill size={20}/> Drug Discovery</NavLink>
          </nav>
        </aside>
        <main className="content">
          <header className="header"><Bell size={22} style={{cursor:'pointer'}}/><div style={{width:35, height:35, borderRadius:'50%', background:'#475569', display:'flex', alignItems:'center', justifyContent:'center'}}><User size={20}/></div></header>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/patients" element={<Patients />} />
            <Route path="/appointments" element={<Appointments />} />
            <Route path="/ai-analysis" element={<AIAnalysis />} />
            <Route path="/drug-discovery" element={<DrugDiscovery />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}