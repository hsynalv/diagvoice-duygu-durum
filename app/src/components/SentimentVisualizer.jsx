import React from 'react';
import Tooltip from './Tooltip';

const SentimentVisualizer = ({ sentiment, score }) => {
  if (sentiment === null || score === null) {
    return null;
  }

  const getSentimentStyle = () => {
    let color;
    let emoji;
    let bgClass;

    if (sentiment.toLowerCase() === 'positive') {
      color = '#059669';
      emoji = '😊';
      bgClass = 'sentiment-positive';
    } else if (sentiment.toLowerCase() === 'negative') {
      color = '#DC2626';
      emoji = '😞';
      bgClass = 'sentiment-negative';
    } else {
      color = '#78716C';
      emoji = '😐';
      bgClass = 'sentiment-neutral';
    }
    return { color, emoji, bgClass };
  };
  
  const { color, emoji, bgClass } = getSentimentStyle();
  const percentage = (score * 100).toFixed(1);

  return (
    <div className={`card sentiment-card ${bgClass}`}>
      <h2>Analiz Sonucu</h2>
      <div className="sentiment-content">
        <span className="sentiment-emoji">{emoji}</span>
        <p className="sentiment-label" style={{ color }}>
          {sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}
        </p>
        <p className="sentiment-score">
          Skor:{' '}
          <Tooltip text="Metin duygu analizinin güven skoru. Yüzde ne kadar yüksekse, model o kadar emin.">
            {percentage}%
          </Tooltip>
        </p>
      </div>
    </div>
  );
};

export default SentimentVisualizer;
