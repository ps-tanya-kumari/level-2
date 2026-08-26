import React, { useState, useEffect, useRef } from 'react';
import { 
  Search, GitFork, Star, Calendar, RefreshCw, Layers, GitMerge, Folder, 
  FileCode, Play, HelpCircle, ChevronRight, CornerDownRight, MessageSquare, 
  User, BookOpen, AlertCircle, CheckCircle, ArrowRight
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8000';

// Recursive File Tree Node Component
const FileTreeNode = ({ node, onSelectFile, selectedPath }) => {
  const [expanded, setExpanded] = useState(false);
  const isDir = node.type === 'tree';
  
  if (isDir) {
    return (
      <div className="tree-node">
        <div 
          className="tree-row" 
          onClick={() => setExpanded(!expanded)}
          style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', padding: '6px 4px' }}
        >
          <span style={{ fontSize: '0.75rem', width: '12px', color: '#6B7280' }}>
            {expanded ? '▼' : '▶'}
          </span>
          <Folder size={16} style={{ color: '#818CF8', flexShrink: 0 }} />
          <span style={{ fontSize: '0.9rem', color: '#E5E7EB' }}>{node.name}</span>
        </div>
        {expanded && node.children && (
          <div className="tree-children" style={{ paddingLeft: '14px', borderLeft: '1px dashed rgba(255,255,255,0.06)' }}>
            {node.children.map((child, i) => (
              <FileTreeNode 
                key={i} 
                node={child} 
                onSelectFile={onSelectFile} 
                selectedPath={selectedPath} 
              />
            ))}
          </div>
        )}
      </div>
    );
  } else {
    const isSelected = selectedPath === node.path;
    return (
      <div 
        className={`tree-row ${isSelected ? 'selected' : ''}`} 
        onClick={() => onSelectFile(node)}
        style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '8px', 
          cursor: 'pointer', 
          padding: '6px 8px', 
          borderRadius: '4px',
          backgroundColor: isSelected ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
          color: isSelected ? '#A5B4FC' : '#9CA3AF',
          transition: 'all 0.15s'
        }}
      >
        <span style={{ width: '12px' }}></span>
        <FileCode size={16} style={{ flexShrink: 0, color: isSelected ? '#A5B4FC' : '#6B7280' }} />
        <span style={{ fontSize: '0.85rem' }}>{node.name}</span>
      </div>
    );
  }
};

export default function App() {
  // Navigation states
  const [username, setUsername] = useState('');
  const [profile, setProfile] = useState(null);
  const [repos, setRepos] = useState([]);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Selected Repository states
  const [selectedRepo, setSelectedRepo] = useState(null);
  const [indexingStatus, setIndexingStatus] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [fileStructure, setFileStructure] = useState(null);
  
  // Tab control
  const [activeTab, setActiveTab] = useState('overview');
  
  // Selected File details (for structure tab)
  const [selectedFile, setSelectedFile] = useState(null);
  const [loadingFileDetail, setLoadingFileDetail] = useState(false);

  // Chat histories
  const [guideHistory, setGuideHistory] = useState([
    { role: 'assistant', content: 'Hi there! I am your AI Project Guide. Select a quick prompt from the sidebar, or ask me any question to understand this codebase step-by-step.' }
  ]);
  const [guideInput, setGuideInput] = useState('');
  const [sendingGuide, setSendingGuide] = useState(false);

  const [qaHistory, setQaHistory] = useState([
    { role: 'assistant', content: 'Ask me anything about the codebase. I will search the repository contents using RAG and give you precise answers citing exact files.' }
  ]);
  const [qaInput, setQaInput] = useState('');
  const [sendingQA, setSendingQA] = useState(false);

  // Scroll references
  const guideEndRef = useRef(null);
  const qaEndRef = useRef(null);

  // Auto-scroll chats
  useEffect(() => {
    guideEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [guideHistory]);

  useEffect(() => {
    qaEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [qaHistory]);

  // Search GitHub Username
  const handleSearchUser = async (e) => {
    e.preventDefault();
    if (!username.trim()) return;
    
    setLoadingProfile(true);
    setErrorMsg('');
    setProfile(null);
    setRepos([]);
    setSelectedRepo(null);
    setAnalysisData(null);
    setFileStructure(null);

    try {
      // 1. Fetch Profile
      const profRes = await fetch(`${API_BASE}/api/github/user/${username}`);
      if (!profRes.ok) throw new Error(`User @${username} not found.`);
      const profData = await profRes.json();
      setProfile(profData);

      // 2. Fetch Repos
      const reposRes = await fetch(`${API_BASE}/api/github/user/${username}/repos`);
      if (reposRes.ok) {
        const reposData = await reposRes.json();
        setRepos(reposData);
      }
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoadingProfile(false);
    }
  };

  // Select a Repository and trigger ingestion
  const handleSelectRepo = async (repo) => {
    setSelectedRepo(repo);
    setAnalysisData(null);
    setFileStructure(null);
    setSelectedFile(null);
    setActiveTab('overview');
    
    // Clear chat histories on repo switch
    setGuideHistory([
      { role: 'assistant', content: `Welcome to the Project Guide for **${repo.name}**! Ask me anything about this repository.` }
    ]);
    setQaHistory([
      { role: 'assistant', content: `Ask me anything about **${repo.name}**. I will fetch matching files via RAG.` }
    ]);

    try {
      // Request backend to index and analyze repository
      const res = await fetch(`${API_BASE}/api/github/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ owner: repo.owner, repo: repo.name })
      });
      const data = await res.json();
      setIndexingStatus(data.status);
    } catch (err) {
      console.error(err);
      setIndexingStatus({ status: 'failed', progress: 0, message: 'Failed to connect to backend service.' });
    }
  };

  // Poll indexing status if in progress
  useEffect(() => {
    let interval = null;
    if (selectedRepo && indexingStatus && indexingStatus.status === 'indexing') {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/api/github/analyze/status/${selectedRepo.owner}/${selectedRepo.name}`);
          const data = await res.json();
          setIndexingStatus(data);
          
          if (data.status === 'completed') {
            // Load Analysis and structure once indexing succeeds
            fetchRepoAnalysisAndStructure(selectedRepo.owner, selectedRepo.name);
            clearInterval(interval);
          } else if (data.status === 'failed') {
            clearInterval(interval);
          }
        } catch (err) {
          console.error("Error polling analysis status:", err);
        }
      }, 2000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [selectedRepo, indexingStatus]);

  // Fetch Analysis (overview/workflow) and folder structure
  const fetchRepoAnalysisAndStructure = async (owner, repoName) => {
    try {
      // Get Overview/Workflow
      const analysisRes = await fetch(`${API_BASE}/api/repository/analysis/${owner}/${repoName}`);
      if (analysisRes.ok) {
        const analysisData = await analysisRes.json();
        setAnalysisData(analysisData);
      }
      
      // Get File Tree Structure
      const structureRes = await fetch(`${API_BASE}/api/repository/structure/${owner}/${repoName}`);
      if (structureRes.ok) {
        const structureData = await structureRes.json();
        setFileStructure(structureData);
      }
    } catch (err) {
      console.error("Error loading analyzed repository data:", err);
    }
  };

  // Explain file selection in the explorer
  const handleSelectFile = async (node) => {
    setLoadingFileDetail(true);
    setSelectedFile({ path: node.path, content: '', explanation: '' });
    
    try {
      const res = await fetch(`${API_BASE}/api/repository/file-detail?owner=${selectedRepo.owner}&repo=${selectedRepo.name}&path=${encodeURIComponent(node.path)}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedFile({
          path: node.path,
          content: data.content,
          explanation: data.explanation
        });
      } else {
        setSelectedFile({
          path: node.path,
          content: 'Could not load file content.',
          explanation: 'Failed to generate explanation for this file.'
        });
      }
    } catch (err) {
      console.error(err);
      setSelectedFile({
        path: node.path,
        content: 'Error loading file content.',
        explanation: 'API connection issue.'
      });
    } finally {
      setLoadingFileDetail(false);
    }
  };

  // Project Guide chat send
  const handleSendGuide = async (customPrompt = null) => {
    const promptToSend = customPrompt || guideInput;
    if (!promptToSend.trim() || sendingGuide) return;

    const newHistory = [...guideHistory, { role: 'user', content: promptToSend }];
    setGuideHistory(newHistory);
    setGuideInput('');
    setSendingGuide(true);

    try {
      const res = await fetch(`${API_BASE}/api/agents/guide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner: selectedRepo.owner,
          repo: selectedRepo.name,
          chat_history: newHistory
        })
      });
      const data = await res.json();
      setGuideHistory([...newHistory, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setGuideHistory([...newHistory, { role: 'assistant', content: 'Connection issue. Could not reach the Project Guide Agent.' }]);
    } finally {
      setSendingGuide(false);
    }
  };

  // Q&A chat send (RAG grounded query)
  const handleSendQA = async () => {
    if (!qaInput.trim() || sendingQA) return;

    const newHistory = [...qaHistory, { role: 'user', content: qaInput }];
    setQaHistory(newHistory);
    setQaInput('');
    setSendingQA(true);

    try {
      const res = await fetch(`${API_BASE}/api/agents/qa`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner: selectedRepo.owner,
          repo: selectedRepo.name,
          chat_history: newHistory
        })
      });
      const data = await res.json();
      setQaHistory([...newHistory, { role: 'assistant', content: data.response }]);
    } catch (err) {
      setQaHistory([...newHistory, { role: 'assistant', content: 'Connection issue. Could not reach the Q&A Agent.' }]);
    } finally {
      setSendingQA(false);
    }
  };

  // Render Workflow dynamic flow diagram
  const renderWorkflowFlow = () => {
    if (!analysisData || !analysisData.workflow || !analysisData.workflow.diagram) return null;
    const rawDiagram = analysisData.workflow.diagram;
    const nodes = rawDiagram.split(/\s*(?:->|→)\s*/);
    
    return (
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '12px', padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid var(--border-muted)', marginBottom: '20px' }}>
        {nodes.map((node, i) => (
          <React.Fragment key={i}>
            <div className="glass-panel" style={{ padding: '12px 20px', borderRadius: '8px', borderLeft: '3px solid var(--primary)', background: '#111827', boxShadow: 'none' }}>
              <span style={{ fontWeight: '600', fontSize: '0.9rem', color: '#F3F4F6' }}>{node}</span>
            </div>
            {i < nodes.length - 1 && (
              <ChevronRight size={20} style={{ color: 'var(--primary)', opacity: 0.8 }} />
            )}
          </React.Fragment>
        ))}
      </div>
    );
  };

  return (
    <div className="container">
      {/* 1. Header Area */}
      <header className="header">
        <div>
          <h1 className="title-gradient" style={{ fontSize: '2rem', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Layers size={32} style={{ color: '#6366F1' }} />
            GitHub Multi-Agent Project Analyzer
          </h1>
          <p className="subtitle">Connect profiles and analyze repository architectures recursively using RAG and Gemini</p>
        </div>
        {selectedRepo && (
          <button 
            className="btn btn-secondary" 
            onClick={() => { setSelectedRepo(null); setAnalysisData(null); }}
          >
            Switch Repository
          </button>
        )}
      </header>

      {/* 2. Main Body Conditional Layouts */}
      {!selectedRepo ? (
        // GitHub Search Profile & Repository Grid Page
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Profile Search input */}
          <form onSubmit={handleSearchUser} className="glass-panel" style={{ padding: '2rem', display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <div style={{ position: 'relative', flex: 1 }}>
              <Search size={20} style={{ position: 'absolute', left: '16px', top: '15px', color: 'var(--text-secondary)' }} />
              <input 
                type="text" 
                className="input-field" 
                placeholder="Enter friend's GitHub username (e.g. torvalds, gaearon)..." 
                style={{ paddingLeft: '48px' }}
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <button type="submit" className="btn btn-primary" style={{ padding: '15px 30px' }} disabled={loadingProfile}>
              {loadingProfile ? <RefreshCw className="animate-spin" size={18} /> : 'Connect profile'}
            </button>
          </form>

          {/* Error Message */}
          {errorMsg && (
            <div className="glass-panel" style={{ padding: '1rem', borderLeft: '4px solid var(--danger)', display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(239, 68, 68, 0.05)' }}>
              <AlertCircle size={20} style={{ color: 'var(--danger)' }} />
              <span style={{ fontSize: '0.95rem' }}>{errorMsg}</span>
            </div>
          )}

          {/* Connection Result Profile Cards */}
          {profile && (
            <div style={{ display: 'grid', gridTemplateColumns: '350px 1fr', gap: '2rem', alignItems: 'flex-start' }}>
              {/* Profile Card */}
              <div className="glass-panel" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', gap: '1rem' }}>
                <img 
                  src={profile.avatar_url} 
                  alt={profile.name} 
                  style={{ width: '120px', height: '120px', borderRadius: '50%', border: '3px solid var(--primary)', boxShadow: 'var(--shadow-neon)' }} 
                />
                <div>
                  <h2 style={{ fontSize: '1.4rem', fontWeight: '700' }}>{profile.name}</h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>@{profile.username}</p>
                </div>
                {profile.bio && <p style={{ fontSize: '0.9rem', color: '#9CA3AF', lineHeight: '1.5' }}>{profile.bio}</p>}
                
                <div style={{ display: 'flex', gap: '1rem', width: '100%', borderTop: '1px solid var(--border-muted)', paddingTop: '1.2rem', marginTop: '0.5rem' }}>
                  <div style={{ flex: 1 }}>
                    <h4 style={{ fontSize: '1.2rem', fontWeight: '600' }}>{profile.public_repos}</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Repositories</p>
                  </div>
                  <div style={{ flex: 1, borderLeft: '1px solid var(--border-muted)', borderRight: '1px solid var(--border-muted)' }}>
                    <h4 style={{ fontSize: '1.2rem', fontWeight: '600' }}>{profile.followers}</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Followers</p>
                  </div>
                  <div style={{ flex: 1 }}>
                    <h4 style={{ fontSize: '1.2rem', fontWeight: '600' }}>{profile.following}</h4>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Following</p>
                  </div>
                </div>
              </div>

              {/* Repositories grid */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <h3 style={{ fontSize: '1.25rem', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <GitFork size={20} style={{ color: 'var(--primary)' }} />
                  Public Repositories
                </h3>
                {repos.length === 0 ? (
                  <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                    No public repositories found.
                  </div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.2rem' }}>
                    {repos.map((repo) => (
                      <div key={repo.name} className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', gap: '1rem' }}>
                        <div>
                          <h4 style={{ fontSize: '1.1rem', fontWeight: '600', color: '#F3F4F6' }}>{repo.name}</h4>
                          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.4rem', lineClamp: 2, WebkitLineClamp: 2, display: '-webkit-box', WebkitBoxOrient: 'vertical', overflow: 'hidden', height: '40px' }}>
                            {repo.description || 'No description provided.'}
                          </p>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
                            <span className="badge badge-primary">{repo.language}</span>
                            <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                              <Star size={14} style={{ color: 'var(--warning)', fill: 'var(--warning)' }} />
                              {repo.stars}
                            </span>
                            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px', marginLeft: 'auto' }}>
                              <Calendar size={14} />
                              {new Date(repo.updated_at).toLocaleDateString()}
                            </span>
                          </div>
                          <button 
                            className="btn btn-primary" 
                            style={{ width: '100%', padding: '10px' }}
                            onClick={() => handleSelectRepo(repo)}
                          >
                            Analyze Project
                            <ArrowRight size={16} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      ) : (
        // Repository Selected Workspace View
        <div>
          {/* Indexing / Processing State Page */}
          {(!analysisData || !fileStructure) ? (
            <div className="glass-panel loader-container">
              {indexingStatus && indexingStatus.status === 'failed' ? (
                <>
                  <AlertCircle size={48} style={{ color: 'var(--danger)', marginBottom: '1rem' }} />
                  <h2>Repository Analysis Failed</h2>
                  <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>{indexingStatus.message}</p>
                  <button className="btn btn-primary" style={{ marginTop: '1.5rem' }} onClick={() => handleSelectRepo(selectedRepo)}>
                    Retry analysis
                  </button>
                </>
              ) : (
                <>
                  <RefreshCw className="animate-spin" size={48} style={{ color: 'var(--primary)', marginBottom: '1.5rem' }} />
                  <h2>Indexing Repository Contents</h2>
                  <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem' }}>Building RAG mappings and embedding files in ChromaDB...</p>
                  
                  <div className="progress-bar-bg">
                    <div 
                      className="progress-bar-fill" 
                      style={{ width: `${indexingStatus ? indexingStatus.progress : 0}%` }}
                    ></div>
                  </div>
                  
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    {indexingStatus ? indexingStatus.message : 'Starting indexing pipeline...'} ({indexingStatus ? indexingStatus.progress : 0}%)
                  </span>
                </>
              )}
            </div>
          ) : (
            // Full 5-Tab Workspace View
            <div className="workspace-layout">
              {/* Left Sidebar Menu */}
              <aside className="glass-panel" style={{ padding: '1.2rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                <div style={{ borderBottom: '1px solid var(--border-muted)', paddingBottom: '1rem', marginBottom: '0.5rem' }}>
                  <h3 style={{ fontSize: '1rem', fontWeight: '700', color: '#FFF' }}>{selectedRepo.name}</h3>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>owner: @{selectedRepo.owner}</p>
                </div>
                
                <div className="sidebar-menu">
                  <div className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
                    <BookOpen size={18} />
                    Project Overview
                  </div>
                  <div className={`tab-btn ${activeTab === 'workflow' ? 'active' : ''}`} onClick={() => setActiveTab('workflow')}>
                    <Play size={18} />
                    Project Workflow
                  </div>
                  <div className={`tab-btn ${activeTab === 'structure' ? 'active' : ''}`} onClick={() => setActiveTab('structure')}>
                    <Folder size={18} />
                    Project Structure
                  </div>
                  <div className={`tab-btn ${activeTab === 'guide' ? 'active' : ''}`} onClick={() => setActiveTab('guide')}>
                    <User size={18} />
                    AI Project Guide
                  </div>
                  <div className={`tab-btn ${activeTab === 'qa' ? 'active' : ''}`} onClick={() => setActiveTab('qa')}>
                    <MessageSquare size={18} />
                    Ask Questions
                  </div>
                </div>
              </aside>

              {/* Main Content Area */}
              <main className="glass-panel" style={{ padding: '2rem', overflowY: 'auto' }}>
                {/* TAB 1: OVERVIEW */}
                {activeTab === 'overview' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    <div style={{ borderBottom: '1px solid var(--border-muted)', paddingBottom: '1rem' }}>
                      <h2 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Project Overview</h2>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.2rem' }}>AI-generated structural analysis of the codebase</p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                      <div>
                        <h3 style={{ fontSize: '1.05rem', fontWeight: '600', marginBottom: '0.5rem', color: '#818CF8' }}>Description</h3>
                        <p style={{ fontSize: '0.95rem', color: '#D1D5DB', lineHeight: '1.6' }}>{analysisData.overview.description}</p>
                      </div>
                      <div>
                        <h3 style={{ fontSize: '1.05rem', fontWeight: '600', marginBottom: '0.5rem', color: '#818CF8' }}>Problem Solved</h3>
                        <p style={{ fontSize: '0.95rem', color: '#D1D5DB', lineHeight: '1.6' }}>{analysisData.overview.problem_solved}</p>
                      </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                      <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#818CF8' }}>Technologies & Stack</h3>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {analysisData.overview.languages.map((l) => <span key={l} className="badge badge-success">{l}</span>)}
                        {analysisData.overview.technologies.map((t) => <span key={t} className="badge badge-primary">{t}</span>)}
                      </div>
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                      <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#818CF8' }}>Important Dependencies</h3>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {analysisData.overview.dependencies.map((d) => (
                          <span key={d} style={{ fontSize: '0.8rem', padding: '4px 10px', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-muted)', borderRadius: '6px', color: '#E5E7EB' }}>{d}</span>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#818CF8', marginBottom: '0.8rem' }}>Key Features</h3>
                      <ul style={{ paddingLeft: '1.5rem', color: '#D1D5DB', lineHeight: '1.8', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {analysisData.overview.features.map((f, i) => <li key={i}>{f}</li>)}
                      </ul>
                    </div>

                    <div>
                      <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#818CF8', marginBottom: '0.8rem' }}>Crucial Files</h3>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        {analysisData.overview.important_files.map((file, i) => (
                          <div key={i} style={{ padding: '12px 16px', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '8px' }}>
                            <code style={{ color: '#A5B4FC', fontWeight: '600' }}>{file.path}</code>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>{file.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#818CF8', marginBottom: '0.5rem' }}>How it Works</h3>
                      <p style={{ fontSize: '0.95rem', color: '#D1D5DB', lineHeight: '1.6' }}>{analysisData.overview.explanation}</p>
                    </div>
                  </div>
                )}

                {/* TAB 2: WORKFLOW */}
                {activeTab === 'workflow' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    <div style={{ borderBottom: '1px solid var(--border-muted)', paddingBottom: '1rem' }}>
                      <h2 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Project Workflow</h2>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.2rem' }}>Dynamically-mapped end-to-end execution flow of the codebase</p>
                    </div>

                    <div>
                      <h3 style={{ fontSize: '1.05rem', fontWeight: '600', color: '#818CF8', marginBottom: '0.8rem' }}>Visual Execution Flow</h3>
                      {renderWorkflowFlow()}
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                      <div style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '12px' }}>
                        <h4 style={{ fontWeight: '600', color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                          <CornerDownRight size={16} /> Starting Point
                        </h4>
                        <p style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.5' }}>{analysisData.workflow.starting_point}</p>
                      </div>

                      <div style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '12px' }}>
                        <h4 style={{ fontWeight: '600', color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                          <CornerDownRight size={16} /> Frontend-to-Backend
                        </h4>
                        <p style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.5' }}>{analysisData.workflow.frontend_backend}</p>
                      </div>

                      <div style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '12px' }}>
                        <h4 style={{ fontWeight: '600', color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                          <CornerDownRight size={16} /> API flow
                        </h4>
                        <p style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.5' }}>{analysisData.workflow.api_flow}</p>
                      </div>

                      <div style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '12px' }}>
                        <h4 style={{ fontWeight: '600', color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                          <CornerDownRight size={16} /> Database interaction
                        </h4>
                        <p style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.5' }}>{analysisData.workflow.database_interaction}</p>
                      </div>
                    </div>

                    <div style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '12px' }}>
                      <h4 style={{ fontWeight: '600', color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                        <CornerDownRight size={16} /> Core Execution Steps
                      </h4>
                      <p style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.6' }}>{analysisData.workflow.execution_flow}</p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                      <div style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '12px' }}>
                        <h4 style={{ fontWeight: '600', color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                          <CornerDownRight size={16} /> Processing Details
                        </h4>
                        <p style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.5' }}>{analysisData.workflow.processing_steps}</p>
                      </div>

                      <div style={{ padding: '1.2rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-muted)', borderRadius: '12px' }}>
                        <h4 style={{ fontWeight: '600', color: '#A5B4FC', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                          <CornerDownRight size={16} /> Final Output
                        </h4>
                        <p style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.5' }}>{analysisData.workflow.final_output}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 3: PROJECT STRUCTURE */}
                {activeTab === 'structure' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
                    <div style={{ borderBottom: '1px solid var(--border-muted)', paddingBottom: '1rem' }}>
                      <h2 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Project Structure</h2>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.2rem' }}>Interactive repository file tree explorer. Select a file to explain with AI.</p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '1.5rem', height: 'calc(100% - 70px)' }}>
                      {/* Left: Tree container */}
                      <div className="glass-panel" style={{ padding: '12px', overflowY: 'auto', background: 'rgba(0,0,0,0.1)' }}>
                        {fileStructure && fileStructure.children && fileStructure.children.map((node, i) => (
                          <FileTreeNode 
                            key={i} 
                            node={node} 
                            onSelectFile={handleSelectFile} 
                            selectedPath={selectedFile ? selectedFile.path : ''} 
                          />
                        ))}
                      </div>

                      {/* Right: File details view */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', height: '100%', overflowY: 'auto' }}>
                        {!selectedFile ? (
                          <div className="glass-panel" style={{ padding: '4rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', color: 'var(--text-muted)', height: '100%' }}>
                            <FileCode size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
                            <h3>No File Selected</h3>
                            <p style={{ fontSize: '0.85rem', marginTop: '0.4rem' }}>Select any source file in the left explorer tree to read content and generate detailed AI explanations.</p>
                          </div>
                        ) : loadingFileDetail ? (
                          <div className="glass-panel" style={{ padding: '4rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', height: '100%' }}>
                            <RefreshCw className="animate-spin" size={32} style={{ color: 'var(--primary)', marginBottom: '1rem' }} />
                            <h3>Fetching File Details</h3>
                            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>Reading content and invoking Gemini analysis for {selectedFile.path}...</p>
                          </div>
                        ) : (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                            {/* File Metadata Info */}
                            <div className="glass-panel" style={{ padding: '1rem 1.5rem', background: '#111827' }}>
                              <h3 style={{ fontSize: '1rem', fontWeight: '700', color: '#F3F4F6' }}>{selectedFile.path.split('/').pop()}</h3>
                              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>Path: <code style={{ color: '#A5B4FC' }}>{selectedFile.path}</code></p>
                            </div>

                            {/* AI Explanation Pane */}
                            <div className="glass-panel" style={{ padding: '1.5rem', borderLeft: '4px solid var(--primary)' }}>
                              <h4 style={{ fontWeight: '700', marginBottom: '8px', color: '#FFF', fontSize: '0.95rem' }}>AI File Explanation</h4>
                              <div style={{ fontSize: '0.9rem', color: '#D1D5DB', lineHeight: '1.6', whiteSpace: 'pre-line' }}>
                                {selectedFile.explanation}
                              </div>
                            </div>

                            {/* Raw Code block */}
                            <div>
                              <h4 style={{ fontWeight: '700', marginBottom: '6px', color: '#FFF', fontSize: '0.95rem' }}>Source Code</h4>
                              <pre style={{ maxHeight: '400px', overflowY: 'auto' }}>
                                <code>{selectedFile.content}</code>
                              </pre>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 4: AI PROJECT GUIDE */}
                {activeTab === 'guide' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
                    <div style={{ borderBottom: '1px solid var(--border-muted)', paddingBottom: '1rem' }}>
                      <h2 style={{ fontSize: '1.5rem', fontWeight: '700' }}>AI Project Guide Agent</h2>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.2rem' }}>Personalized step-by-step developer onboarding agent. Click a preset topic or chat directly.</p>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '1.5rem', height: 'calc(100% - 70px)' }}>
                      {/* Left Sidebar Preset Prompts */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                        <h4 style={{ fontSize: '0.85rem', fontWeight: '700', textTransform: 'uppercase', tracking: '0.05em', color: 'var(--text-muted)', marginBottom: '4px' }}>Quick Start Topics</h4>
                        
                        <button className="btn btn-secondary" style={{ padding: '10px 14px', fontSize: '0.8rem', justifyContent: 'flex-start', textAlign: 'left', borderRadius: '8px', fontWeight: '500' }} onClick={() => handleSendGuide('Explain this project to me like a beginner.')}>
                          Explain like a beginner
                        </button>
                        <button className="btn btn-secondary" style={{ padding: '10px 14px', fontSize: '0.8rem', justifyContent: 'flex-start', textAlign: 'left', borderRadius: '8px', fontWeight: '500' }} onClick={() => handleSendGuide('Where should I start reading this project?')}>
                          Where do I start reading?
                        </button>
                        <button className="btn btn-secondary" style={{ padding: '10px 14px', fontSize: '0.8rem', justifyContent: 'flex-start', textAlign: 'left', borderRadius: '8px', fontWeight: '500' }} onClick={() => handleSendGuide('Explain the backend directory structure and architecture.')}>
                          Explain the backend
                        </button>
                        <button className="btn btn-secondary" style={{ padding: '10px 14px', fontSize: '0.8rem', justifyContent: 'flex-start', textAlign: 'left', borderRadius: '8px', fontWeight: '500' }} onClick={() => handleSendGuide('Explain the frontend directory structure.')}>
                          Explain the frontend
                        </button>
                        <button className="btn btn-secondary" style={{ padding: '10px 14px', fontSize: '0.8rem', justifyContent: 'flex-start', textAlign: 'left', borderRadius: '8px', fontWeight: '500' }} onClick={() => handleSendGuide('Which file serves as the main entry point, and explain its code.')}>
                          What is the entry point?
                        </button>
                      </div>

                      {/* Right Chat Panel */}
                      <div className="chat-container">
                        <div className="chat-messages">
                          {guideHistory.map((msg, i) => (
                            <div key={i} className={`chat-bubble ${msg.role}`}>
                              <span style={{ fontSize: '0.95rem', whiteSpace: 'pre-line' }}>{msg.content}</span>
                            </div>
                          ))}
                          {sendingGuide && (
                            <div className="chat-bubble assistant" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <RefreshCw className="animate-spin" size={16} />
                              <span>Guide Agent is reasoning and executing tools...</span>
                            </div>
                          )}
                          <div ref={guideEndRef}></div>
                        </div>

                        <div className="chat-input-wrapper">
                          <input 
                            type="text" 
                            className="input-field" 
                            placeholder="Ask the Guide Agent a step-by-step question..."
                            value={guideInput}
                            onChange={(e) => setGuideInput(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter') handleSendGuide(); }}
                            disabled={sendingGuide}
                          />
                          <button className="btn btn-primary" onClick={() => handleSendGuide()} disabled={sendingGuide}>
                            Send
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 5: ASK QUESTIONS (Q&A) */}
                {activeTab === 'qa' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
                    <div style={{ borderBottom: '1px solid var(--border-muted)', paddingBottom: '1rem' }}>
                      <h2 style={{ fontSize: '1.5rem', fontWeight: '700' }}>Q&A Agent (RAG Grounded Search)</h2>
                      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.2rem' }}>Ask specific code questions. The agent fetches similar code blocks from ChromaDB (RAG) and explains referencing exact file paths.</p>
                    </div>

                    <div className="chat-container" style={{ height: 'calc(100% - 70px)' }}>
                      <div className="chat-messages">
                        {qaHistory.map((msg, i) => (
                          <div key={i} className={`chat-bubble ${msg.role}`}>
                            <span style={{ fontSize: '0.95rem', whiteSpace: 'pre-line' }}>{msg.content}</span>
                          </div>
                        ))}
                        {sendingQA && (
                          <div className="chat-bubble assistant" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <RefreshCw className="animate-spin" size={16} />
                            <span>Q&A Agent is searching vector database (RAG) and generating answer...</span>
                          </div>
                        )}
                        <div ref={qaEndRef}></div>
                      </div>

                      <div className="chat-input-wrapper">
                        <input 
                          type="text" 
                          className="input-field" 
                          placeholder="Search codebase (e.g. 'How does authentication work?', 'Where is DB connected?')..."
                          value={qaInput}
                          onChange={(e) => setQaInput(e.target.value)}
                          onKeyDown={(e) => { if (e.key === 'Enter') handleSendQA(); }}
                          disabled={sendingQA}
                        />
                        <button className="btn btn-primary" onClick={handleSendQA} disabled={sendingQA}>
                          Ask RAG
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </main>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
