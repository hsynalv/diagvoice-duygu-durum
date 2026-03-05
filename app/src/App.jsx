import { useState } from 'react';
import './App.css';
import AudioRecorder from './components/AudioRecorder';
import SampleAudioPlayer from './components/SampleAudioPlayer';
import AudioFilePicker from './components/AudioFilePicker';
import SentimentVisualizer from './components/SentimentVisualizer';
import Tooltip from './components/Tooltip';
import { analyzeFused } from './services/analyzeFused';
import logo from './logo.jpg';

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
  const [mentalFitnessResult, setMentalFitnessResult] = useState(null);
  const [ageGenderResult, setAgeGenderResult] = useState(null);
  const [valenceResult, setValenceResult] = useState({ text: null, audio: null, fused: null });
  const [confidenceResult, setConfidenceResult] = useState(null);
  const [error, setError] = useState('');
  const [currentAnalysisSource, setCurrentAnalysisSource] = useState('');

  const handleAudioAnalysis = async (audioBlob, sourceName) => {
    // Reset state
    setIsLoading(true);
    setError('');
    setTranscribedText('');
    setSentimentResult({ sentiment: null, score: null });
    setAudioEmotionResult(null);
    setDiseaseResult(null);
    setMentalFitnessResult(null);
    setAgeGenderResult(null);
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

      if (result.mental_fitness) {
        setMentalFitnessResult(result.mental_fitness);
      }

      if (result.age_gender) {
        setAgeGenderResult(result.age_gender);
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
        <h1 className="title">Duygu Durum Analizi</h1>
      </header>
      
      <div className="main-grid">
        <AudioRecorder onRecordingComplete={(blob) => handleAudioAnalysis(blob, 'Yeni Kayıt')} />
        <SampleAudioPlayer onSampleSelect={handleAudioAnalysis} />
        <AudioFilePicker onFileSelect={handleAudioAnalysis} />
      </div>

      {isLoading && <div className="loading">"{currentAnalysisSource}" için analiz yapılıyor... Lütfen bekleyin.</div>}
      
      {error && <div className="error">{error}</div>}

      <section className="results-section">
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
              <h2>Ses Duygusu</h2>
            </div>
            <div className="main-result">
              <span className="result-highlight" style={{ color: getEmotionColor(audioEmotionResult.pred_label) }}>
                {getEmotionLabel(audioEmotionResult.pred_label)}
              </span>
              <span className="confidence-badge">
                Güven: {Math.round((audioEmotionResult.intensity || 0) * 100)}%
              </span>
            </div>
            <div className="prob-section">
              <h4>Olasılıklar</h4>
              {Array.isArray(audioEmotionResult.probs) && audioEmotionResult.probs.map((prob, idx) => {
                const labels = ['Üzüntü', 'Korku', 'Mutluluk', 'Öfke'];
                const colors = ['#6366f1', '#8b5cf6', '#10b981', '#ef4444'];
                return <ProbBar key={idx} label={labels[idx]} value={prob} color={colors[idx]} />;
              })}
            </div>
          </div>
        )}

        {diseaseResult && (
          <div className="card health-card">
            <div className="card-header">
              <div className={`health-indicator ${diseaseResult.pred_label === 'healthy' ? 'healthy' : 'sick'}`}></div>
              <h2>Sağlık Durumu</h2>
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
              <h2>Ses Sağlığı</h2>
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

        {(valenceResult.text !== null || valenceResult.audio !== null || valenceResult.fused !== null) && (
          <div className="card fusion-card">
            <div className="card-header">
              <div className="fusion-indicator"></div>
              <h2>Genel Duygu Değerlendirmesi</h2>
            </div>
            <div className="fusion-result">
              <div className={`fusion-label ${valenceResult.fused?.label === 'positive' ? 'positive' : valenceResult.fused?.label === 'negative' ? 'negative' : 'neutral'}`}>
                {valenceResult.fused?.label === 'positive' ? 'Olumlu' : valenceResult.fused?.label === 'negative' ? 'Olumsuz' : 'Nötr'}
              </div>
              <div className="fusion-confidence">
                Güven: {Math.round((valenceResult.fused?.valence || 0) * 100)}%
              </div>
            </div>
            <div className="fusion-sources">
              <div className="source-bar">
                <span className="source-label">Metin</span>
                <div className="source-bar-container">
                  <div className="source-fill text-source" style={{ width: `${Math.round((valenceResult.text || 0) * 100)}%` }}></div>
                </div>
                <span className="source-value">{Math.round((valenceResult.text || 0) * 100)}%</span>
              </div>
              <div className="source-bar">
                <span className="source-label">Ses</span>
                <div className="source-bar-container">
                  <div className="source-fill audio-source" style={{ width: `${Math.round((valenceResult.audio || 0) * 100)}%` }}></div>
                </div>
                <span className="source-value">{Math.round((valenceResult.audio || 0) * 100)}%</span>
              </div>
            </div>
            {confidenceResult?.dynamic_fusion && (
              <div className="fusion-note">
                Metin ve ses ağırlığı otomatik ayarlandı
              </div>
            )}
          </div>
        )}
        
        {ageGenderResult && (
          <div className="card profile-card">
            <div className="card-header">
              <div className="profile-indicator"></div>
              <h2>Kişi Profili</h2>
            </div>
            <div className="profile-grid">
              <div className="profile-item">
                <span className="profile-label">Cinsiyet</span>
                <span className="profile-value" style={{ color: GENDER_LABELS[ageGenderResult.gender?.pred_label]?.color || '#374151' }}>
                  {GENDER_LABELS[ageGenderResult.gender?.pred_label]?.tr || ageGenderResult.gender?.pred_label || '-'}
                </span>
                <div className="profile-confidence">
                  {ageGenderResult.gender?.probs && (
                    <div className="mini-bars">
                      <div className="mini-bar" style={{ width: `${Math.round(ageGenderResult.gender.probs[0] * 100)}%`, backgroundColor: '#3b82f6' }}></div>
                      <div className="mini-bar" style={{ width: `${Math.round(ageGenderResult.gender.probs[1] * 100)}%`, backgroundColor: '#ec4899' }}></div>
                    </div>
                  )}
                </div>
              </div>
              <div className="profile-item">
                <span className="profile-label">Yaş Aralığı</span>
                <span className="profile-value">{ageGenderResult.agebin?.pred_label || '-'}</span>
              </div>
            </div>
          </div>
        )}
        
        {sentimentResult.sentiment && (
          <SentimentVisualizer 
            sentiment={sentimentResult.sentiment} 
            score={sentimentResult.score} 
          />
        )}
      </section>
    </div>
  );
}

export default App;
