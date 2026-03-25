import { useState, useEffect } from 'react';

const AudioFilePicker = ({ onFileSelect }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleChange = (e) => {
    const file = e.target.files?.[0] ?? null;
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    setSelectedFile(file);
    if (file) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      onFileSelect(file, file.name);
    }
  };

  return (
    <div className="card card-file-picker">
      <h2>Bilgisayardan ses</h2>
      <input
        type="file"
        accept="audio/*"
        onChange={handleChange}
      />
      {selectedFile && (
        <>
          <div className="result-row" style={{ borderBottom: 'none', paddingTop: '0.75rem' }}>
            <span className="result-label">Seçilen</span>
            <span className="result-value">{selectedFile.name}</span>
          </div>
          {previewUrl && (
            <div className="file-preview-audio-wrap">
              <audio
                className="file-preview-audio"
                controls
                src={previewUrl}
                preload="metadata"
              >
                Tarayıcınız ses oynatmayı desteklemiyor.
              </audio>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default AudioFilePicker;
