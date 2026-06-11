import { useState, useRef } from "react";

export default function FileUpload({
  onFileSelect,
  file,
  maxSentences,
  onMaxSentencesChange,
  totalSentences,
  onAnalyze,
  isAnalyzing,
}) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => setDragOver(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) onFileSelect(droppedFile);
  };

  const handleChange = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) onFileSelect(selectedFile);
  };

  const clearFile = (e) => {
    e.stopPropagation();
    onFileSelect(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const zoneClasses = [
    "upload-zone",
    dragOver && "drag-over",
    file && "has-file",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className="upload-section" id="upload-section">
      <h2 className="section-header">
        <span className="icon">📄</span>
        Upload an SRS Document
      </h2>

      <div
        className={zoneClasses}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".txt,.pdf"
          onChange={handleChange}
          id="file-upload-input"
        />

        {!file ? (
          <>
            <div className="upload-icon">📁</div>
            <div className="upload-text">
              Drag & drop your file here, or click to browse
            </div>
            <div className="upload-hint">Supports .txt and .pdf files</div>
          </>
        ) : (
          <>
            <div className="upload-icon">✅</div>
            <div className="upload-text">File selected</div>
            <div className="file-info">
              <span className="file-name">{file.name}</span>
              <button
                className="file-clear"
                onClick={clearFile}
                title="Remove file"
              >
                ✕
              </button>
            </div>
          </>
        )}
      </div>

      {file && totalSentences > 0 && (
        <>
          <div className="status-message status-info">
            📊 Extracted <strong>{totalSentences} sentences</strong> from{" "}
            <strong>{file.name}</strong>
          </div>

          <div className="slider-group">
            <span className="slider-label">Max sentences to analyze:</span>
            <input
              type="range"
              className="slider-input"
              min={5}
              max={Math.min(200, totalSentences)}
              value={maxSentences}
              onChange={(e) => onMaxSentencesChange(Number(e.target.value))}
              id="max-sentences-slider"
            />
            <span className="slider-value">{maxSentences}</span>
          </div>

          <button
            className="analyze-btn"
            onClick={onAnalyze}
            disabled={isAnalyzing}
            id="analyze-btn"
          >
            {isAnalyzing ? (
              <>
                <span className="spinner" />
                Analyzing…
              </>
            ) : (
              <>🔍 Analyze Document</>
            )}
          </button>
        </>
      )}
    </section>
  );
}
