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

export default function ModelDetailsPage({
  ageGenderResult,
  ageGenderError,
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

  const renderAgeGender = () => {
    if (!gender && !agebin) return null;
    const genderPred = gender?.pred_label;
    const agebinPred = agebin?.pred_label;

    const genderColor =
      genderPred && GENDER_LABELS[genderPred] ? GENDER_LABELS[genderPred].color : '#3b82f6';

    const agebinColor = '#f59e0b';

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
                {agebinPred || '-'}
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
                label={GENDER_LABELS[gender.pred_label]?.tr || gender.pred_label || 'Tahmin'}
                value={genderTop2[0]?.p ?? genderBest.value}
                color={genderColor}
              />
              {genderTop2.length > 1 && (
                <ProbBarRow
                  label={gender.probs.length === 2 ? 'Diğer' : `Sınıf ${genderTop2[1].i + 1}`}
                  value={genderTop2[1].p}
                  color="#94a3b8"
                />
              )}
            </>
          )}

          {Array.isArray(agebin?.probs) && agebin.probs.length > 0 && (
            <>
              <ProbBarRow
                label={agebinPred || 'Tahmin'}
                value={agebinTop2[0]?.p ?? agebinBest.value}
                color={agebinColor}
              />
              {agebinTop2.length > 1 && (
                <ProbBarRow
                  label={agebin.probs.length === 2 ? 'Diğer' : `Sınıf ${agebinTop2[1].i + 1}`}
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
          <h2>Metin vs Ses Valence</h2>
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
          <h2>Güven & Dinamik Füzyon</h2>
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
    <div>
      {renderAgeGender()}
      {renderValence()}
      {renderValenceBars()}
      {renderConfidence()}
    </div>
  );
}

