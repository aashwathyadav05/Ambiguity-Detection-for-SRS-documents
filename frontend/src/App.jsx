import { useState, useEffect } from "react";
import { checkHealth, analyzeDocument } from "./api/client";
import Hero from "./components/Hero";
import LabelGuide from "./components/LabelGuide";
import FileUpload from "./components/FileUpload";
import DocumentSummary from "./components/DocumentSummary";
import SentenceResult from "./components/SentenceResult";
import Footer from "./components/Footer";
import "./index.css";

export default function App() {
  // Backend status
  const [backendStatus, setBackendStatus] = useState(null); // null = loading, object = loaded
  const [backendError, setBackendError] = useState(null);

  // File & analysis state
  const [file, setFile] = useState(null);
  const [maxSentences, setMaxSentences] = useState(50);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  // Derived: total sentences is only known after analysis
  // We'll show the slider once analysis is done or use a default
  const totalSentences = analysisResult?.total_sentences ?? 0;

  // Check backend health on mount
  useEffect(() => {
    checkHealth()
      .then((data) => setBackendStatus(data))
      .catch((err) => setBackendError(err.message));
  }, []);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setAnalysisResult(null);
    setAnalysisError(null);
    setMaxSentences(50);
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    try {
      const result = await analyzeDocument(file, maxSentences);
      setAnalysisResult(result);
    } catch (err) {
      setAnalysisError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="app-container">
      <Hero />

      {/* Backend status */}
      {backendError && (
        <div className="status-message status-error">
          ⚠️ Backend not reachable: {backendError}. Make sure the FastAPI server
          is running on <code>localhost:8000</code>.
        </div>
      )}
      {backendStatus && (
        <div
          className={`status-message ${
            backendStatus.model_loaded ? "status-success" : "status-warning"
          }`}
        >
          {backendStatus.model_loaded
            ? `✓ Model loaded from ${backendStatus.model_path}`
            : `⚠ Model not found at ${backendStatus.model_path}. Rule-based heuristics are still available.`}
        </div>
      )}

      <LabelGuide />

      <FileUpload
        file={file}
        onFileSelect={handleFileSelect}
        maxSentences={maxSentences}
        onMaxSentencesChange={setMaxSentences}
        totalSentences={analysisResult ? analysisResult.total_sentences : 200}
        onAnalyze={handleAnalyze}
        isAnalyzing={isAnalyzing}
      />

      {/* Analysis error */}
      {analysisError && (
        <div className="status-message status-error">
          ❌ {analysisError}
        </div>
      )}

      {/* Analysis results */}
      {analysisResult && (
        <>
          <DocumentSummary
            summary={analysisResult.summary}
            totalSentences={analysisResult.total_sentences}
            analyzedSentences={analysisResult.analyzed_sentences}
            filename={analysisResult.filename}
          />

          <section className="results-section" id="sentence-results">
            <h2 className="section-header">
              <span className="icon">🔬</span>
              Sentence-level Results
            </h2>
            {analysisResult.results.map((result, index) => (
              <SentenceResult
                key={index}
                result={result}
                index={index}
              />
            ))}
          </section>
        </>
      )}

      <Footer />
    </div>
  );
}
