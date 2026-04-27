import { useState } from 'react';
import './App.css';
import AudioRecorder from './components/AudioRecorder';
import SampleAudioPlayer from './components/SampleAudioPlayer';
import AudioFilePicker from './components/AudioFilePicker';
// import SentimentVisualizer from './components/SentimentVisualizer';
import { analyzeFused } from './services/analyzeFused';
import logo from './logo.jpg';
// Geçici: sadece özet skorlar — model detayları sekmesi kapalı
// import ModelDetailsPage from './components/ModelDetailsPage';

// Geçici: WURSS / emosyon kartları kapalıyken yorumda
// function ProbBar({ label, value, color }) { ... }
// const EMOTION_LABELS = { ... }
// const GENDER_LABELS = { ... }
// function getEmotionLabel(pred_label) { ... }
// function getEmotionColor(pred_label) { ... }

/** İş Sağlığı benchmark özet kartı: yalnızca arayüz; P(tanılı) ≥ bu oran → "Tanılı riski" */
const BENCHMARK_POSITIVE_UI_THRESHOLD = 0.5;

const DEPRESSION_RISK_TR = { dusuk: 'Düşük', orta: 'Orta', yuksek: 'Yüksek' };
const DEPRESSION_RISK_COLOR = { dusuk: '#10b981', orta: '#f59e0b', yuksek: '#ef4444' };

function App() {
  const [isLoading, setIsLoading] = useState(false);
  // Alt çizgi: arayüzde geçici gizlendi, setter'lar hâlâ doldurulur
  const [_transcribedText, setTranscribedText] = useState('');
  const [_sentimentResult, setSentimentResult] = useState({ sentiment: null, score: null });
  const [_audioEmotionResult, setAudioEmotionResult] = useState(null);
  const [_diseaseResult, setDiseaseResult] = useState(null);
  const [depressionResult, setDepressionResult] = useState(null);
  const [_mentalFitnessResult, setMentalFitnessResult] = useState(null);
  const [_ageGenderResult, setAgeGenderResult] = useState(null);
  const [_ageGenderError, setAgeGenderError] = useState(null);
  const [_valenceResult, setValenceResult] = useState({ text: null, audio: null, fused: null });
  const [_confidenceResult, setConfidenceResult] = useState(null);
  const [benchmarkV2Result, setBenchmarkV2Result] = useState(null);
  const [benchmarkV2Error, setBenchmarkV2Error] = useState(null);
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
    setBenchmarkV2Result(null);
    setBenchmarkV2Error(null);
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

      if (result.benchmark_v2) {
        setBenchmarkV2Result(result.benchmark_v2);
      }
      if (typeof result.benchmark_v2_error === 'string') {
        setBenchmarkV2Error(result.benchmark_v2_error);
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
        {/* Geçici: model detayları sekmesi kapalı
        <button
          className={`tab-button ${activePage === 'details' ? 'active' : 'secondary'}`}
          onClick={() => setActivePage('details')}
          type="button"
          disabled={isLoading}
        >
          Model Detayları
        </button>
        */}
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
            {/* Geçici: sadece İş Sağlığı + Depresyon skorları — diğer kartlar kapalı
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
            */}

            {/* Geçici: WURSS (hastalık) kartı kapalı */}

            {(benchmarkV2Result || benchmarkV2Error) && (
              <div className="card health-card">
                <div className="card-header">
                  <div
                    className="profile-indicator"
                    style={{
                      background: (() => {
                        if (benchmarkV2Error || benchmarkV2Result?.error) return '#94a3b8';
                        const p = benchmarkV2Result?.positive_class_probability;
                        if (typeof p !== 'number') return '#94a3b8';
                        return p >= BENCHMARK_POSITIVE_UI_THRESHOLD ? '#f59e0b' : '#10b981';
                      })(),
                    }}
                  />
                  <h2>İş Sağlığı</h2>
                </div>
                {benchmarkV2Error && (
                  <div className="error" style={{ marginTop: 0 }}>
                    {benchmarkV2Error}
                  </div>
                )}
                {benchmarkV2Result?.error && (
                  <div className="error" style={{ marginTop: 0 }}>
                    {typeof benchmarkV2Result.error === 'string'
                      ? benchmarkV2Result.error
                      : JSON.stringify(benchmarkV2Result.error)}
                  </div>
                )}
                {benchmarkV2Result &&
                  typeof benchmarkV2Result.positive_class_probability === 'number' &&
                  !benchmarkV2Result.error &&
                  (() => {
                    const pTan = benchmarkV2Result.positive_class_probability ?? 0;
                    const uiTanili = pTan >= BENCHMARK_POSITIVE_UI_THRESHOLD;
                    return (
                      <>
                        <div className="main-result">
                          <span
                            className={`result-highlight ${uiTanili ? 'sick-text' : 'healthy-text'}`}
                          >
                            {uiTanili ? 'Tanılı riski' : 'Sağlıklı profil'}
                          </span>
                          <span className="confidence-badge">
                            Eşik: %{Math.round(BENCHMARK_POSITIVE_UI_THRESHOLD * 100)}
                          </span>
                        </div>
                        <div className="score-section">
                          <div className="score-label">P(tanılı)</div>
                          <div className="score-bar-container">
                            <div
                              className="score-bar"
                              style={{
                                width: `${Math.max(0, Math.min(100, pTan * 100))}%`,
                                backgroundColor:
                                  pTan >= BENCHMARK_POSITIVE_UI_THRESHOLD ? '#f59e0b' : '#10b981',
                              }}
                            />
                          </div>
                          <div className="score-value">%{Math.round(pTan * 100)}</div>
                        </div>
                        {benchmarkV2Result.mode && (
                          <div className="profile-confidence" style={{ marginTop: '0.5rem' }}>
                            Model modu: {benchmarkV2Result.mode}
                          </div>
                        )}
                      </>
                    );
                  })()}
              </div>
            )}

            {/* Geçici: mental fitness gizlendi
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
            */}

            {depressionResult && (() => {
              const rl = depressionResult.risk_level;
              const riskTr = (rl && DEPRESSION_RISK_TR[rl]) || '—';
              const riskColor = (rl && DEPRESSION_RISK_COLOR[rl]) || '#6b7280';
              const depressionStatus =
                depressionResult.pred_label === 'depresyon' ? 'Depresyon Riski' : 'Sağlıklı';
              return (
                <div className="card health-card">
                  <div className="card-header">
                    <div className="profile-indicator" style={{ background: riskColor }}></div>
                    <h2>Depresyon</h2>
                  </div>
                  <div className="main-result" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.25rem' }}>
                    <span className="result-highlight" style={{ color: riskColor }}>
                      Durum: {depressionStatus}
                    </span>
                    <span className="result-highlight" style={{ color: riskColor }}>
                      Risk seviyesi: {riskTr}
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Geçici: metin duygusu gizlendi
            {sentimentResult.sentiment && (
              <SentimentVisualizer
                sentiment={sentimentResult.sentiment}
                score={sentimentResult.score}
              />
            )}
            */}
          </>
        )}

        {/* Geçici: model detayları sayfası kapalı
        {activePage === 'details' && (
          <ModelDetailsPage
            audioEmotionResult={audioEmotionResult}
            ageGenderResult={ageGenderResult}
            ageGenderError={ageGenderError}
            valenceResult={valenceResult}
            confidenceResult={confidenceResult}
          />
        )}
        */}
      </section>
    </div>
  );
}

export default App;
