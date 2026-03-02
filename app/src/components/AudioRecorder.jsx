import React, { useState, useRef, useEffect, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import RecordPlugin from 'wavesurfer.js/dist/plugins/record.esm.js';

const AudioRecorder = ({ onRecordingComplete }) => {
  const waveContainerRef = useRef(null);
  const wavesurferRef = useRef(null);
  const recordPluginRef = useRef(null);
  
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState('');

  const createWaveSurfer = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.destroy();
    }
    return WaveSurfer.create({
      container: waveContainerRef.current,
      waveColor: '#A8A8A8',
      progressColor: '#0D9488',
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      height: 80,
      cursorWidth: 1,
      cursorColor: 'transparent',
    });
  }, []);

  const startRecording = useCallback(async () => {
    setAudioUrl('');
    wavesurferRef.current = createWaveSurfer();
    
    const record = wavesurferRef.current.registerPlugin(RecordPlugin.create({ scrollingWaveform: true }));
    recordPluginRef.current = record;

    record.on('record-end', (blob) => {
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      onRecordingComplete(blob);
      record.destroy();
      wavesurferRef.current.load(url);
    });
    
    // Make the record plugin listen to the progress of the wavesurfer
    wavesurferRef.current.on('timeupdate', (currentTime) => {
        if(isRecording) record.render(currentTime)
    });

    try {
      await record.startRecording();
      setIsRecording(true);
    } catch(e) {
      console.error('Error starting recording:', e);
      alert('Mikrofon erişimi reddedildi veya bir hata oluştu.');
    }
  }, [createWaveSurfer, onRecordingComplete, isRecording]);

  const stopRecording = useCallback(() => {
    if (recordPluginRef.current) {
      recordPluginRef.current.stopRecording();
      setIsRecording(false);
    }
  }, []);
  
  const handlePlayPause = () => {
      if(wavesurferRef.current) {
          wavesurferRef.current.playPause();
          setIsPlaying(wavesurferRef.current.isPlaying());
      }
  }

  useEffect(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.on('play', () => setIsPlaying(true));
      wavesurferRef.current.on('pause', () => setIsPlaying(false));
    }
  }, [audioUrl]);

  return (
    <div className="card">
      <h2>Yeni Ses Kaydı Yap</h2>
      <div ref={waveContainerRef} className="waveform-container"></div>
      <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
        {!isRecording ? (
          <button onClick={startRecording} disabled={isRecording}>
            Kaydı Başlat
          </button>
        ) : (
          <button onClick={stopRecording} disabled={!isRecording}>
            Kaydı Durdur
          </button>
        )}
        {audioUrl && !isRecording && (
          <button onClick={handlePlayPause} className="secondary">
            {isPlaying ? 'Durdur' : 'Oynat'}
          </button>
        )}
      </div>
    </div>
  );
};

export default AudioRecorder;
