import React, { useState, useCallback, useRef } from 'react'
import Plasma from './plasma.jsx';

export default function App(){
  const [uploadResult, setUploadResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const uploadRef = useRef(null);

  return (
    <div style={{ width: '100%', height: '100%', position: 'absolute' }}>
      <Plasma 
        color="#5effdccc"
        speed={0.6}
        direction="forward"
        scale={0.9}
        opacity={0.85}
        mouseInteractive={true}
      />
      <div className="header">
        <h1 className="title">Welcome.</h1>
        <p className="sub">Upload images of fruits & vegetables and we'll classify them.</p>
        <button
          className="get-started"
          onClick={() => uploadRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })}
        >
          Get started
        </button>
      </div>
      <div className="foreground-card">
        <div id="upload-section" ref={uploadRef}>
          <UploadBox onSubmit={setUploadResult} loading={loading} setLoading={setLoading} />
          {uploadResult && (
            <div className="results-section">
              <h3>Classification Results:</h3>
              {uploadResult.results.map((result, idx) => (
                <div className="result" key={idx}>
                  <strong>{result.filename}</strong>: {result.prediction} 
                  <span className="confidence"> ({(result.confidence * 100).toFixed(2)}% confidence)</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function UploadBox({ onSubmit, loading, setLoading }){
  const [files, setFiles] = useState([]);
  
  const onFiles = useCallback((incoming) => {
    const arr = Array.from(incoming).slice(0, 5);
    setFiles(arr.map(f => Object.assign(f, { preview: URL.createObjectURL(f) })));
  }, []);

  const handleDrop = e => {
    e.preventDefault();
    onFiles(e.dataTransfer.files);
  };
  
  const handlePick = e => onFiles(e.target.files);

  const handleSubmit = async () => {
    if (!files.length) return;
    
    setLoading(true);
    const fd = new FormData();
    files.forEach((f, i) => fd.append('file' + i, f));
    
    try {
    
      const res = await fetch('http://localhost:8000/predict', { 
        method: 'POST', 
        body: fd 
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      onSubmit && onSubmit(data);
    } catch (err) {
      alert('Upload failed: ' + err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-box" onDrop={handleDrop} onDragOver={e => e.preventDefault()}>
      <input 
        id="file-picker" 
        type="file" 
        accept="image/*" 
        multiple 
        onChange={handlePick} 
        style={{display:'none'}} 
      />
      <label htmlFor="file-picker" className="pick-btn">Choose files</label>
      <div className="upload-hint">drag & upload images here</div>
      <div className="previews">
        {files.map((f, idx) => (
          <div className="preview" key={idx}>
            <img src={f.preview} alt={f.name} />
            <div className="fname">{f.name}</div>
          </div>
        ))}
      </div>
      <button 
        className="submit-btn" 
        onClick={handleSubmit} 
        disabled={!files.length || loading}
      >
        {loading ? 'Classifying...' : 'Upload & Classify'}
      </button>
    </div>
  );
}