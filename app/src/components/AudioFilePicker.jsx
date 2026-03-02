import React, { useState } from 'react';

const AudioFilePicker = ({ onFileSelect }) => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleChange = (e) => {
    const file = e.target.files && e.target.files[0] ? e.target.files[0] : null;
    setSelectedFile(file);
    if (file) {
      onFileSelect(file, file.name);
    }
  };

  return (
    <div className="card">
      <h2>Bilgisayardan Ses Seç</h2>
      <input
        type="file"
        accept="audio/*"
        onChange={handleChange}
      />
      {selectedFile && (
        <div className="result-row" style={{ borderBottom: 'none', paddingTop: '0.75rem' }}>
          <span className="result-label">Seçilen</span>
          <span className="result-value">{selectedFile.name}</span>
        </div>
      )}
    </div>
  );
};

export default AudioFilePicker;
