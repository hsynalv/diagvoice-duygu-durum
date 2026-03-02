import { useState } from 'react';
import './App.css';
import AudioRecorder from './components/AudioRecorder';
import SampleAudioPlayer from './components/SampleAudioPlayer';
import AudioFilePicker from './components/AudioFilePicker';
import SentimentVisualizer from './components/SentimentVisualizer';
import Tooltip from './components/Tooltip';
import { analyzeFused } from './services/analyzeFused';
import logo from './logo.jpg';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [sentimentResult, setSentimentResult] = useState({ sentiment: null, score: null });
  const [audioEmotionResult, setAudioEmotionResult] = useState(null);
  const [diseaseResult, setDiseaseResult] = useState(null);
  const [mentalFitnessResult, setMentalFitnessResult] = useState(null);
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
          <div className="card">
            <h2>Ses Üzerinden Duygu</h2>
            <div className="result-row">
              <span className="result-label">Tahmin</span>
              <span className="result-value">
                <Tooltip text="Ses tonundan çıkarılan duygu (üzüntü, korku, mutluluk, öfke)">
                  {audioEmotionResult.pred_label} (id: {audioEmotionResult.pred_id})
                </Tooltip>
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Intensity</span>
              <span className="result-value">
                <Tooltip text="Duygunun ses tonundaki yoğunluğu. 0 ile 1 arası; yüksek değer duygunun daha belirgin olduğunu gösterir.">
                  {typeof audioEmotionResult.intensity === 'number' ? audioEmotionResult.intensity.toFixed(3) : audioEmotionResult.intensity}
                </Tooltip>
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Olasılıklar</span>
              <span className="result-value">
                <Tooltip text="Üzüntü, korku, mutluluk, öfke sınıfları için model tahmin olasılıkları (sırasıyla).">
                  {Array.isArray(audioEmotionResult.probs) ? audioEmotionResult.probs.map((p) => Number(p).toFixed(3)).join(' | ') : ''}
                </Tooltip>
              </span>
            </div>
          </div>
        )}

        {diseaseResult && (
          <div className="card">
            <h2>ÜSYE Tahmini</h2>
            <div className="result-row">
              <span className="result-label">Tahmin</span>
              <span className="result-value">
                <Tooltip text="ÜSYE modeline göre ses kaydından çıkarılan sağlık tahmini (healthy: sağlıklı, sick: hasta).">
                  {diseaseResult.pred_label === 'healthy' ? 'Sağlıklı' : diseaseResult.pred_label === 'sick' ? 'Hasta' : diseaseResult.pred_label} (id: {diseaseResult.pred_id})
                </Tooltip>
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Olasılıklar</span>
              <span className="result-value">
                <Tooltip text="Sağlıklı ve hasta sınıfları için model tahmin olasılıkları (sırasıyla).">
                  {Array.isArray(diseaseResult.probs) ? diseaseResult.probs.map((p) => Number(p).toFixed(3)).join(' | ') : ''}
                </Tooltip>
              </span>
            </div>
          </div>
        )}

        {mentalFitnessResult && (
          <div className="card">
            <h2>Canlıda Olan</h2>
            <div className="result-row">
              <span className="result-label">Tahmin</span>
              <span className="result-value">
                <Tooltip text="Canlıda olan modeline göre ses kaydından çıkarılan tahmin.">
                  {mentalFitnessResult.pred_label ?? JSON.stringify(mentalFitnessResult)}
                  {typeof mentalFitnessResult.pred_id === 'number' ? ` (id: ${mentalFitnessResult.pred_id})` : ''}
                </Tooltip>
              </span>
            </div>
            {Array.isArray(mentalFitnessResult.probs) && (
              <div className="result-row">
                <span className="result-label">Olasılıklar</span>
                <span className="result-value">
                  <Tooltip text="Canlıda olan modelinin sınıfları için tahmin olasılıkları.">
                    {mentalFitnessResult.probs.map((p) => Number(p).toFixed(3)).join(' | ')}
                  </Tooltip>
                </span>
              </div>
            )}
            {typeof mentalFitnessResult.mental_fitness_score === 'number' && (
              <div className="result-row">
                <span className="result-label">Canlılık Skoru</span>
                <span className="result-value">
                  <Tooltip text="0–100 arası; yüksek değer sesin 'sağlıklı' profile daha yakın olduğunu gösterir.">
                    {mentalFitnessResult.mental_fitness_score.toFixed(2)}%
                  </Tooltip>
                </span>
              </div>
            )}
          </div>
        )}

        {(valenceResult.text !== null || valenceResult.audio !== null || valenceResult.fused !== null) && (
          <div className="card">
            <h2>Füzyon (Text + Ses)</h2>
            <div className="result-row">
              <span className="result-label">Text Valence</span>
              <span className="result-value">
                <Tooltip text="Metin duygu analizinden gelen değer. 0 ile 1 arası; 0 olumsuz, 1 olumlu.">
                  {typeof valenceResult.text === 'number' ? valenceResult.text.toFixed(3) : '-'}
                </Tooltip>
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Audio Valence</span>
              <span className="result-value">
                <Tooltip text="Ses duygu analizinden gelen değer. 0 ile 1 arası; 0 olumsuz, 1 olumlu.">
                  {typeof valenceResult.audio === 'number' ? valenceResult.audio.toFixed(3) : '-'}
                </Tooltip>
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">Fused</span>
              <span className="result-value">
                <Tooltip text="Metin ve sesin birleştirilmiş sonucu. Dinamik ağırlıkla hesaplanır.">
                  {valenceResult.fused?.label ?? '-'} ({typeof valenceResult.fused?.valence === 'number' ? valenceResult.fused.valence.toFixed(3) : '-'})
                </Tooltip>
              </span>
            </div>
            <div className="result-row">
              <span className="result-label">w_text</span>
              <span className="result-value">
                <Tooltip text="Metin değerine verilen ağırlık (0–1). Yüksek değer metin duygusunun füzyonda daha güçlü etkili olduğunu gösterir.">
                  {typeof valenceResult.fused?.w_text === 'number' ? valenceResult.fused.w_text.toFixed(3) : (valenceResult.fused?.w_text ?? '-')}
                </Tooltip>
              </span>
            </div>
            {confidenceResult && (
              <>
                <div className="result-row">
                  <span className="result-label">Text Güven</span>
                  <span className="result-value">
                    <Tooltip text="Metin duygu analizinin güveni. 0–1 arası; yüksek değer modelin daha emin olduğunu gösterir.">
                      {typeof confidenceResult.text === 'number' ? confidenceResult.text.toFixed(3) : '-'}
                    </Tooltip>
                  </span>
                </div>
                <div className="result-row">
                  <span className="result-label">Audio Güven</span>
                  <span className="result-value">
                    <Tooltip text="Ses duygu analizinin güveni. 0–1 arası; yüksek değer modelin daha emin olduğunu gösterir.">
                      {typeof confidenceResult.audio === 'number' ? confidenceResult.audio.toFixed(3) : '-'}
                    </Tooltip>
                  </span>
                </div>
                <div className="result-row">
                  <span className="result-label">Uyuşmazlık</span>
                  <span className="result-value">
                    <Tooltip text="Metin ve ses valence değerleri arasındaki fark. Yüksek değer iki kaynağın birbiriyle çeliştiğini gösterir.">
                      {typeof confidenceResult.disagreement === 'number' ? confidenceResult.disagreement.toFixed(3) : '-'}
                    </Tooltip>
                  </span>
                </div>
                <div className="result-row">
                  <span className="result-label">Dinamik Füzyon</span>
                  <span className="result-value">
                    <Tooltip text="Güven skorlarına göre w_text ağırlığının otomatik ayarlanıp ayarlanmadığı.">
                      {confidenceResult.dynamic_fusion ? 'Açık' : 'Kapalı'}
                    </Tooltip>
                  </span>
                </div>
              </>
            )}
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
