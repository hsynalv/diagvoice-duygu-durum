import { useState } from 'react';
import test1 from '../assets/audio/test1.m4a';
import test2 from '../assets/audio/test2.m4a';
import test3 from '../assets/audio/test3.wav';
import test4 from '../assets/audio/test4.wav';
import test5 from '../assets/audio/test5.wav';
import test6 from '../assets/audio/test6.wav';

const sampleAudios = [
  { name: 'Test 1', path: test1, fileName: 'test1.m4a' },
  { name: 'Test 2', path: test2, fileName: 'test2.m4a' },
  { name: 'Test 3', path: test3, fileName: 'test3.wav' },
  { name: 'Test 4', path: test4, fileName: 'test4.wav' },
  { name: 'Test 5', path: test5, fileName: 'test5.wav' },
  { name: 'Test 6', path: test6, fileName: 'test6.wav' },
];

const SampleAudioPlayer = ({ onSampleSelect }) => {
  const [active, setActive] = useState(sampleAudios[0]);

  const handleSelect = async (audio) => {
    try {
      const response = await fetch(audio.path);
      const blob = await response.blob();
      const type = blob.type && blob.type !== 'application/octet-stream'
        ? blob.type
        : (audio.fileName.endsWith('.m4a') ? 'audio/mp4' : 'audio/wav');
      const file = new File([blob], audio.fileName, { type });
      onSampleSelect(file, audio.name);
    } catch (error) {
      console.error('Error fetching the audio sample:', error);
      alert('Örnek ses dosyası yüklenirken bir hata oluştu.');
    }
  };

  return (
    <div className="card card-sample">
      <h2>Örnek sesler</h2>
      <p className="sample-picker-hint">
        Bir örnek seçin; aynı alandan dinleyip analizi başlatabilirsiniz.
      </p>
      <div className="sample-chip-grid" role="listbox" aria-label="Örnek ses listesi">
        {sampleAudios.map((audio) => {
          const isActive = active.fileName === audio.fileName;
          return (
            <button
              key={audio.fileName}
              type="button"
              role="option"
              aria-selected={isActive}
              className={`sample-chip${isActive ? ' is-active' : ''}`}
              onClick={() => setActive(audio)}
            >
              {audio.name}
            </button>
          );
        })}
      </div>
      <div className="sample-player-panel">
        <audio
          key={active.fileName}
          className="sample-shared-audio"
          controls
          src={active.path}
          preload="metadata"
        >
          Tarayıcınız ses oynatmayı desteklemiyor.
        </audio>
        <button
          type="button"
          className="sample-analyze-btn"
          onClick={() => handleSelect(active)}
        >
          {active.name} — Analiz et
        </button>
      </div>
    </div>
  );
};

export default SampleAudioPlayer;
