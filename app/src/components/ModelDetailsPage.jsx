import React from 'react';

function formatPct01(v, digits = 0) {
  if (typeof v !== 'number' || Number.isNaN(v)) return null;
  return `${(v * 100).toFixed(digits)}%`;
}

function maxIndex(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return { idx: -1, value: null };
  let bestIdx = 0;
  let bestVal = arr[0];
  for (let i = 1; i < arr.length; i += 1) {
    if (arr[i] > bestVal) {
      bestVal = arr[i];
      bestIdx = i;
    }
  }
  return { idx: bestIdx, value: bestVal };
}

function ProbBarRow({ label, value, color }) {
  const percentage = Math.round((value || 0) * 100);
  return (
    <div className="prob-bar-row">
      <span className="prob-label">{label}</span>
      <div className="prob-bar-container">
        <div className="prob-bar-fill" style={{ width: `${percentage}%`, backgroundColor: color }} />
        <span className="prob-value">{percentage}%</span>
      </div>
    </div>
  );
}

const GENDER_LABELS = {
  male: { tr: 'Erkek', color: '#3b82f6' },
  female: { tr: 'Kadın', color: '#ec4899' },
};
const AGEBIN_LABELS = ['15-24', '25-34', '35-44', '45-54', '55-64'];
const AGE_DISPLAY_SHIFT_BINS = 1; // UI'da bir kademe daha genç göster

export default function ModelDetailsPage({
  audioEmotionResult,
  ageGenderResult,
  ageGenderError,
  benchmarkV2Result,
  benchmarkV2Error,
  valenceResult,
  confidenceResult,
}) {
  const gender = ageGenderResult?.gender;
  const agebin = ageGenderResult?.agebin;

  const genderBest = maxIndex(gender?.probs);
  const agebinBest = maxIndex(agebin?.probs);

  const genderTop2 = Array.isArray(gender?.probs)
    ? gender.probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 2)
    : [];
  const agebinTop2 = Array.isArray(agebin?.probs)
    ? agebin.probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 2)
    : [];

  const fused = valenceResult?.fused;
  const fusedValence = typeof fused?.valence === 'number' ? fused.valence : null;
  const fusedLabel = fused?.label;

  const valenceText = typeof valenceResult?.text === 'number' ? valenceResult.text : null;
  const valenceAudio = typeof valenceResult?.audio === 'number' ? valenceResult.audio : null;

  const confidenceText = confidenceResult?.text;
  const confidenceAudio = confidenceResult?.audio;
  const disagreement = confidenceResult?.disagreement;
  const dynamicFusion = confidenceResult?.dynamic_fusion;

  const fusedAccentColor =
    fusedLabel?.toLowerCase() === 'positive' ? '#10b981' : fusedLabel?.toLowerCase() === 'negative' ? '#ef4444' : '#6b7280';
  const emotionLabels = ['Üzüntü', 'Korku', 'Mutluluk', 'Öfke'];
  const emotionColors = ['#6366f1', '#8b5cf6', '#10b981', '#ef4444'];

  const renderBenchmarkV2 = () => {
    if (!benchmarkV2Result && !benchmarkV2Error) return null;

    const errMsg =
      benchmarkV2Error ||
      (benchmarkV2Result?.error
        ? typeof benchmarkV2Result.error === 'string'
          ? benchmarkV2Result.error
          : JSON.stringify(benchmarkV2Result.error)
        : null);

    const b = benchmarkV2Result;
    const ok =
      b &&
      typeof b.positive_class_probability === 'number' &&
      !b.error;

    const predId = typeof b?.predicted_class === 'number' ? b.predicted_class : null;
    const thresh = typeof b?.threshold_tuned === 'number' ? b.threshold_tuned : 0.5;
    const pPos = typeof b?.positive_class_probability === 'number' ? b.positive_class_probability : null;

    return (
      <div className="card health-card" style={{ paddingTop: 18 }}>
        <div className="card-header">
          <div
            className="profile-indicator"
            style={{
              background: errMsg ? '#94a3b8' : predId === 1 ? '#f59e0b' : '#10b981',
            }}
          />
          <h2>Benchmark v2 — tam çıktı</h2>
        </div>
        <div className="profile-confidence" style={{ marginBottom: '0.75rem' }}>
          Kaynak: fusion → iç inference_api (tabular joblib). Sınıf 0/1 eğitim etiketine bağlıdır.
        </div>

        {errMsg && <div className="error" style={{ marginBottom: '1rem' }}>{errMsg}</div>}

        {ok && (
          <>
            <div className="profile-grid">
              <div className="profile-item">
                <div className="profile-label">Tahmin</div>
                <div className={`profile-value ${predId === 1 ? 'sick-text' : 'healthy-text'}`}>
                  {predId === 1 ? 'Tanılı riski' : 'Sağlıklı profil'}
                </div>
                <div className="profile-confidence">predicted_class: {predId}</div>
              </div>
              <div className="profile-item">
                <div className="profile-label">P(tanılı)</div>
                <div className="profile-value">{formatPct01(pPos, 1) || '-'}</div>
                <div className="profile-confidence">Eşik: {formatPct01(thresh, 0) || '-'}</div>
              </div>
              <div className="profile-item">
                <div className="profile-label">Eşik hedefi</div>
                <div className="profile-value">{b.threshold_objective || '-'}</div>
                <div className="profile-confidence">threshold_objective</div>
              </div>
              <div className="profile-item">
                <div className="profile-label">Model modu</div>
                <div className="profile-value">{b.mode || '-'}</div>
                <div className="profile-confidence">bundle.mode</div>
              </div>
            </div>

            {b.class_names_hint && (
              <div className="profile-confidence" style={{ marginBottom: '0.75rem' }}>
                {b.class_names_hint}
              </div>
            )}

            <div className="prob-section">
              <h4>Ham JSON</h4>
              <pre
                style={{
                  margin: 0,
                  padding: '12px',
                  background: 'rgba(0,0,0,0.25)',
                  borderRadius: 8,
                  fontSize: 12,
                  overflow: 'auto',
                  maxHeight: 280,
                }}
              >
                {JSON.stringify(b, null, 2)}
              </pre>
            </div>
          </>
        )}
      </div>
    );
  };

  const renderEmotionDistribution = () => {
    if (!Array.isArray(audioEmotionResult?.probs) || audioEmotionResult.probs.length === 0) return null;

    return (
      <div className="card health-card" style={{ paddingTop: 18 }}>
        <div className="card-header">
          <div className="fusion-indicator" />
          <h2>Ses Duygusu (Emosyon) - Olasılık Dağılımı</h2>
        </div>
        <div className="profile-confidence" style={{ marginBottom: '0.75rem' }}>
          Kaynak: Ses Duygusu (Emosyon) modelinin sınıf bazlı olasılık çıktıları
        </div>
        <div className="prob-section">
          {audioEmotionResult.probs.map((prob, idx) => (
            <ProbBarRow
              key={`${emotionLabels[idx] || `Sınıf ${idx + 1}`}-${idx}`}
              label={emotionLabels[idx] || `Sınıf ${idx + 1}`}
              value={prob}
              color={emotionColors[idx] || '#94a3b8'}
            />
          ))}
        </div>
      </div>
    );
  };

  const renderAgeGender = () => {
    if (!gender && !agebin) return null;
    const genderPred = gender?.pred_label;
    const agebinPred = agebin?.pred_label;
    const genderPredId = typeof gender?.pred_id === 'number' ? gender.pred_id : null;
    const agebinPredId = typeof agebin?.pred_id === 'number' ? agebin.pred_id : null;

    const genderColor =
      genderPred && GENDER_LABELS[genderPred] ? GENDER_LABELS[genderPred].color : '#3b82f6';

    const agebinColor = '#f59e0b';
    const trGender = (label) => GENDER_LABELS[label]?.tr || label || '-';
    const oppositeGender =
      genderPred === 'female' ? 'male' : genderPred === 'male' ? 'female' : null;

    const getGenderLabelByIndex = (idx) => {
      if (genderPredId !== null && idx === genderPredId) return trGender(genderPred);
      if (Array.isArray(gender?.probs) && gender.probs.length === 2 && oppositeGender) return trGender(oppositeGender);
      return `Sınıf ${idx + 1}`;
    };

    const getAgebinLabelByIndex = (idx) => {
      const shiftedIdx = Math.max(0, Math.min(AGEBIN_LABELS.length - 1, idx - AGE_DISPLAY_SHIFT_BINS));
      return AGEBIN_LABELS[shiftedIdx] || `Sınıf ${shiftedIdx + 1}`;
    };
    const agebinPredDisplay = (() => {
      if (agebinPredId !== null) return getAgebinLabelByIndex(agebinPredId);
      const inferredIdx = AGEBIN_LABELS.indexOf(agebinPred || '');
      if (inferredIdx >= 0) return getAgebinLabelByIndex(inferredIdx);
      return agebinPred || '-';
    })();

    return (
      <div className="card wellness-card">
        <div className="card-header">
          <div className="profile-indicator" style={{ background: '#f59e0b' }} />
          <h2>Yaş & Cinsiyet</h2>
        </div>

        {ageGenderError && <div className="error" style={{ marginBottom: '1rem' }}>{ageGenderError}</div>}

        {gender && (
          <div className="profile-grid">
            <div className="profile-item">
              <div className="profile-label">Cinsiyet Tahmini</div>
              <div className="profile-value" style={{ color: genderColor }}>
                {GENDER_LABELS[gender.pred_label]?.tr || gender.pred_label}
              </div>
              {typeof genderBest.value === 'number' && (
                <div className="profile-confidence">
                  Güven: {formatPct01(genderBest.value, 0)}
                </div>
              )}
            </div>

            <div className="profile-item">
              <div className="profile-label">Yaş Grubu Tahmini</div>
              <div className="profile-value" style={{ color: agebinColor }}>
                {agebinPredDisplay}
              </div>
              {typeof agebinBest.value === 'number' && (
                <div className="profile-confidence">
                  Güven: {formatPct01(agebinBest.value, 0)}
                </div>
              )}
            </div>
          </div>
        )}

        <div className="prob-section">
          <h4>Dağılım (Olasılıklar)</h4>
          {Array.isArray(gender?.probs) && gender.probs.length > 0 && (
            <>
              <ProbBarRow
                label={getGenderLabelByIndex(genderTop2[0]?.i ?? genderBest.idx)}
                value={genderTop2[0]?.p ?? genderBest.value}
                color={genderColor}
              />
              {genderTop2.length > 1 && (
                <ProbBarRow
                  label={getGenderLabelByIndex(genderTop2[1].i)}
                  value={genderTop2[1].p}
                  color="#94a3b8"
                />
              )}
            </>
          )}

          {Array.isArray(agebin?.probs) && agebin.probs.length > 0 && (
            <>
              <ProbBarRow
                label={getAgebinLabelByIndex(agebinTop2[0]?.i ?? agebinBest.idx)}
                value={agebinTop2[0]?.p ?? agebinBest.value}
                color={agebinColor}
              />
              {agebinTop2.length > 1 && (
                <ProbBarRow
                  label={getAgebinLabelByIndex(agebinTop2[1].i)}
                  value={agebinTop2[1].p}
                  color="#94a3b8"
                />
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  const renderValence = () => {
    const hasAnyValence = valenceText !== null || valenceAudio !== null || fusedValence !== null;
    if (!hasAnyValence) return null;

    return (
      <div className="card fusion-result">
        <div>
          <div className={`fusion-label ${fusedLabel?.toLowerCase() === 'positive' ? 'positive' : fusedLabel?.toLowerCase() === 'negative' ? 'negative' : 'neutral'}`}>
            Fused Duygu: {fusedLabel ? fusedLabel[0] + fusedLabel.slice(1).toLowerCase() : '-'}
          </div>
          <div className="fusion-confidence">
            {typeof fusedValence === 'number' ? `Valence: ${(fusedValence * 100).toFixed(1)}%` : 'Valence hesaplanamadı'}
          </div>
          {typeof fused?.w_text === 'number' && <div className="fusion-confidence">Metin ağırlığı: {Math.round(fused.w_text * 100)}%</div>}
        </div>

        <div style={{ width: 160 }}>
          <div className="source-bar-container" style={{ height: 14, borderRadius: 10, marginBottom: 8 }}>
            <div
              className="source-fill"
              style={{ width: `${typeof fusedValence === 'number' ? fusedValence * 100 : 0}%`, background: fusedAccentColor, height: '100%' }}
            />
          </div>
          <div className="source-value" style={{ textAlign: 'right' }}>
            {typeof fusedValence === 'number' ? `${(fusedValence * 100).toFixed(0)}%` : '-'}
          </div>
        </div>
      </div>
    );
  };

  const renderValenceBars = () => {
    if (valenceText === null && valenceAudio === null) return null;
    return (
      <div className="card health-card" style={{ paddingTop: 18 }}>
        <div className="card-header">
          <div className="fusion-indicator" />
          <h2>Ses Duygusu (Emosyon) - Metin vs Ses Valence</h2>
        </div>
        <div className="profile-confidence" style={{ marginBottom: '0.75rem' }}>
          Kaynak: metin duygu analizi + ses emosyonundan hesaplanan valence karşılaştırması
        </div>
        <div className="prob-section">
          {valenceText !== null && (
            <ProbBarRow label="Metin" value={valenceText} color="#3b82f6" />
          )}
          {valenceAudio !== null && (
            <ProbBarRow label="Ses" value={valenceAudio} color="#8b5cf6" />
          )}
        </div>
      </div>
    );
  };

  const renderConfidence = () => {
    const has =
      typeof confidenceText === 'number' ||
      typeof confidenceAudio === 'number' ||
      typeof disagreement === 'number';
    if (!has) return null;

    return (
      <div className="card health-card">
        <div className="card-header">
          <div className="fusion-indicator" />
          <h2>Ses Duygusu (Emosyon) - Güven & Dinamik Füzyon</h2>
        </div>
        <div className="profile-confidence" style={{ marginBottom: '0.75rem' }}>
          Kaynak: "Ses Duygusu (Emosyon)" + metin duygu analizi birleştirme
        </div>

        <div className="profile-grid">
          <div className="profile-item">
            <div className="profile-label">Metin Güveni</div>
            <div className="profile-value">{formatPct01(confidenceText, 0) || '-'}</div>
            <div className="profile-confidence">Valence_text’e göre</div>
          </div>
          <div className="profile-item">
            <div className="profile-label">Ses Güveni</div>
            <div className="profile-value">{formatPct01(confidenceAudio, 0) || '-'}</div>
            <div className="profile-confidence">Audio entropy’ya göre</div>
          </div>
        </div>

        <div className="prob-section">
          <h4>Uyumsuzluk & Model Ayarı</h4>
          {typeof disagreement === 'number' && (
            <div className="result-row">
              <span className="result-label">Disagreement</span>
              <span className="result-value">{disagreement.toFixed(3)}</span>
            </div>
          )}
          <div className="result-row">
            <span className="result-label">Dynamic Fusion</span>
            <span className="result-value">{dynamicFusion ? 'Açık' : 'Kapalı'}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="model-details-stack">
      {renderBenchmarkV2()}
      {renderEmotionDistribution()}
      {renderAgeGender()}
      {renderValence()}
      {renderValenceBars()}
      {renderConfidence()}
    </div>
  );
}

