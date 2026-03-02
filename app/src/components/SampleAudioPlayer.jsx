import React from 'react';

// Vite's way of handling static assets
import harikaGun from '../assets/audio/bu-harika-bir-gun.m4a';
import derstenKalacagim from '../assets/audio/dersten-kalacağım.m4a';
import karisik from '../assets/audio/karisik.m4a';

const sampleAudios = [
  { name: "Bu harika bir gün", path: harikaGun },
  { name: "Dersten kalacağım", path: derstenKalacagim },
  { name: "Karışık", path: karisik },
];

const SampleAudioPlayer = ({ onSampleSelect }) => {
  const handleSelect = async (audio) => {
    try {
      const response = await fetch(audio.path);
      const blob = await response.blob();
      onSampleSelect(blob, audio.name);
    } catch (error) {
      console.error("Error fetching the audio sample:", error);
      alert("Örnek ses dosyası yüklenirken bir hata oluştu.");
    }
  };

  return (
    <div className="card">
      <h2>Veya Örnek Bir Ses Seç</h2>
      <div className="sample-list">
        {sampleAudios.map((audio) => (
          <div key={audio.name} className="sample-item">
            <span>{audio.name}</span>
            <button onClick={() => handleSelect(audio)}>Analiz Et</button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SampleAudioPlayer;
