import { useState } from 'react';
import './App.css';
import AudioRecorder from './components/AudioRecorder';
import SampleAudioPlayer from './components/SampleAudioPlayer';
import AudioFilePicker from './components/AudioFilePicker';
import SentimentVisualizer from './components/SentimentVisualizer';
import Tooltip from './components/Tooltip';
import { analyzeFused } from './services/analyzeFused';
import logo from './logo.jpg';
import ModelDetailsPage from './components/ModelDetailsPage';

// Probability bar component
function ProbBar({ label, value, color }) {
  const percentage = Math.round(value * 100);
  return (
    <div className="prob-bar-row">
      <span className="prob-label">{label}</span>
      <div className="prob-bar-container">
        <div 
          className="prob-bar-fill" 
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
        <span className="prob-value">{percentage}%</span>
      </div>
    </div>
  );
}

// Emotion mapping for audio results
const EMOTION_LABELS = {
  'sadness': { tr: 'Üzüntü', color: '#6366f1' },
  'fear': { tr: 'Korku', color: '#8b5cf6' },
  'happiness': { tr: 'Mutluluk', color: '#10b981' },
  'anger': { tr: 'Öfke', color: '#ef4444' },
  'happy': { tr: 'Mutlu', color: '#10b981' },
  'sad': { tr: 'Üzgün', color: '#6366f1' },
  'angry': { tr: 'Sinirli', color: '#ef4444' },
  'fearful': { tr: 'Korkulu', color: '#8b5cf6' }
};

// Gender mapping
const GENDER_LABELS = {
  'male': { tr: 'Erkek', color: '#3b82f6' },
  'female': { tr: 'Kadın', color: '#ec4899' }
};

function getEmotionLabel(pred_label) {
  return EMOTION_LABELS[pred_label]?.tr || pred_label;
}

function getEmotionColor(pred_label) {
  return EMOTION_LABELS[pred_label]?.color || '#6b7280';
}

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [sentimentResult, setSentimentResult] = useState({ sentiment: null, score: null });
  const [audioEmotionResult, setAudioEmotionResult] = useState(null);
  const [diseaseResult, setDiseaseResult] = useState(null);
  const [depressionResult, setDepressionResult] = useState(null);
  const [mentalFitnessResult, setMentalFitnessResult] = useState(null);
  const [ageGenderResult, setAgeGenderResult] = useState(null);
  const [ageGenderError, setAgeGenderError] = useState(null);
  const [valenceResult, setValenceResult] = useState({ text: null, audio: null, fused: null });
  const [confidenceResult, setConfidenceResult] = useState(null);
  const [error, setError] = useState('');
  const [currentAnalysisSource, setCurrentAnalysisSource] = useState('');
  const [activePage, setActivePage] = useState('summary');

  const handleAudioAnalysis = async (audioBlob, sourceName) => {
    // Reset state
    setIsLoading(true);
    setError('');
    setTranscribedText('');
    setSentimentResult({ sentiment: null, score: null });
    setAudioEmotionResult(null);
    setDiseaseResult(null);
    setDepressionResult(null);
    setMentalFitnessResult(null);
    setAgeGenderResult(null);
    setAgeGenderError(null);
    setValenceResult({ text: null, audio: null, fused: null });
    setConfidenceResult(null);
    setCurrentAnalysisSource(sourceName);

    try {
      const result = await analyzeFused(audioBlob);
      setTranscribedText(result.text || '');

      if (result.text_sentiment) {
        setSentimentResult({
          sentiment: result.text_sentiment.sentiment ?? null,
          score: result.text_sentiment.score ?? null,
        });
      }

      if (result.audio) {
        setAudioEmotionResult(result.audio);
      }

      if (result.disease) {
        setDiseaseResult(result.disease);
      }

      if (result.depression) {
        setDepressionResult(result.depression);
      }

      if (result.mental_fitness) {
        setMentalFitnessResult(result.mental_fitness);
      }

      if (result.age_gender) {
        setAgeGenderResult(result.age_gender);
      }
      if (typeof result.age_gender_error === 'string') {
        setAgeGenderError(result.age_gender_error);
      } else {
        setAgeGenderError(null);
      }

      setValenceResult({
        text: result.valence_text ?? null,
        audio: result.valence_audio ?? null,
        fused: result.fused ?? null,
      });

      if (result.confidence) {
        setConfidenceResult(result.confidence);
      }

    } catch (err) {
      const errorMessage = err.message || 'Bilinmeyen bir hata oluştu';
      console.error(`Analysis failed for ${sourceName}:`, err);
      setError(`"${sourceName}" analizi sırasında bir hata oluştu: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <img src={logo} alt="DiagVoice Logo" className="logo" />
        <div className="brand-name">DiagVoice</div>
      </header>

      <div className="page-tabs">
        <button
          className={`tab-button ${activePage === 'summary' ? 'active' : 'secondary'}`}
          onClick={() => setActivePage('summary')}
          type="button"
          disabled={isLoading}
        >
          Özet
        </button>
        <button
          className={`tab-button ${activePage === 'details' ? 'active' : 'secondary'}`}
          onClick={() => setActivePage('details')}
          type="button"
          disabled={isLoading}
        >
          Model Detayları
        </button>
      </div>
      
      <div className="main-grid">
        <AudioRecorder onRecordingComplete={(blob) => handleAudioAnalysis(blob, 'Yeni Kayıt')} />
        <SampleAudioPlayer onSampleSelect={handleAudioAnalysis} />
        <AudioFilePicker onFileSelect={handleAudioAnalysis} />
      </div>

      {isLoading && <div className="loading">"{currentAnalysisSource}" için analiz yapılıyor... Lütfen bekleyin.</div>}
      
      {error && <div className="error">{error}</div>}

      <section className="results-section">
        {activePage === 'summary' && (
          <>
            {transcribedText && (
              <div className="card transcribed-text">
                <h2>Çevrilen Metin</h2>
                <p>"{transcribedText}"</p>
              </div>
            )}

            {audioEmotionResult && (
              <div className="card emotion-card">
                <div className="card-header">
                  <div className="emotion-indicator" style={{ backgroundColor: getEmotionColor(audioEmotionResult.pred_label) }}></div>
                  <h2>Ses Duygusu (Emosyon)</h2>
                </div>
                <div className="main-result">
                  <span className="result-highlight" style={{ color: getEmotionColor(audioEmotionResult.pred_label) }}>
                    {getEmotionLabel(audioEmotionResult.pred_label)}
                  </span>
                  <span className="confidence-badge">
                    Güven: {Math.round((audioEmotionResult.intensity || 0) * 100)}%
                  </span>
                </div>
              </div>
            )}

            {diseaseResult && (
              <div className="card health-card">
                <div className="card-header">
                  <div className={`health-indicator ${diseaseResult.pred_label === 'healthy' ? 'healthy' : 'sick'}`}></div>
                  <h2>WURSS</h2>
                </div>
                <div className="main-result">
                  <span className={`result-highlight ${diseaseResult.pred_label === 'healthy' ? 'healthy-text' : 'sick-text'}`}>
                    {diseaseResult.pred_label === 'healthy' ? 'Sağlıklı' : 'Hasta'}
                  </span>
                </div>
                <div className="prob-section compact">
                  <ProbBar label="Sağlıklı" value={diseaseResult.probs?.[0] || 0} color="#10b981" />
                  <ProbBar label="Hasta" value={diseaseResult.probs?.[1] || 0} color="#ef4444" />
                </div>
              </div>
            )}

            {mentalFitnessResult && (
              <div className="card wellness-card">
                <div className="card-header">
                  <div className="wellness-indicator"></div>
                  <h2>Mental Fitness</h2>
                </div>
                <div className="main-result">
                  <span className="result-highlight">
                    {mentalFitnessResult.pred_label === 'healthy' ? 'Sağlıklı Profil' : 'Değerlendirme Gerekli'}
                  </span>
                </div>
                {typeof mentalFitnessResult.mental_fitness_score === 'number' && (
                  <div className="score-section">
                    <div className="score-label">Ses Sağlığı Skoru</div>
                    <div className="score-bar-container">
                      <div
                        className="score-bar"
                        style={{ width: `${mentalFitnessResult.mental_fitness_score}%`, backgroundColor: mentalFitnessResult.mental_fitness_score > 70 ? '#10b981' : mentalFitnessResult.mental_fitness_score > 40 ? '#f59e0b' : '#ef4444' }}
                      />
                    </div>
                    <div className="score-value">{mentalFitnessResult.mental_fitness_score.toFixed(0)}%</div>
                  </div>
                )}
              </div>
            )}

            {depressionResult && (
              <div className="card health-card">
                <div className="card-header">
                  <div className="profile-indicator" style={{ background: depressionResult.pred_id === 1 ? '#ef4444' : '#10b981' }}></div>
                  <h2>Depresyon Skoru</h2>
                </div>
                <div className="main-result">
                  <span className={`result-highlight ${depressionResult.pred_id === 1 ? 'sick-text' : 'healthy-text'}`}>
                    {depressionResult.pred_label === 'depresyon' ? 'Depresyon Riski' : 'Sağlıklı Profil'}
                  </span>
                  <span className="confidence-badge">
                    Eşik: %{Math.round((depressionResult.threshold ?? 0.5) * 100)}
                  </span>
                </div>
                <div className="score-section">
                  <div className="score-label">Ortalama Depresyon Olasılığı</div>
                  <div className="score-bar-container">
                    <div
                      className="score-bar"
                      style={{
                        width: `${Math.max(0, Math.min(100, (depressionResult.mean_prob_depression || 0) * 100))}%`,
                        backgroundColor: (depressionResult.mean_prob_depression || 0) >= (depressionResult.threshold ?? 0.5) ? '#ef4444' : '#10b981'
                      }}
                    />
                  </div>
                  <div className="score-value">%{Math.round((depressionResult.mean_prob_depression || 0) * 100)}</div>
                </div>
              </div>
            )}

            {sentimentResult.sentiment && (
              <SentimentVisualizer
                sentiment={sentimentResult.sentiment}
                score={sentimentResult.score}
              />
            )}
          </>
        )}

        {activePage === 'details' && (
          <ModelDetailsPage
            audioEmotionResult={audioEmotionResult}
            ageGenderResult={ageGenderResult}
            ageGenderError={ageGenderError}
            depressionResult={depressionResult}
            valenceResult={valenceResult}
            confidenceResult={confidenceResult}
          />
        )}
      </section>
    </div>
  );
}

export default App;
