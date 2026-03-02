import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8003/analyze-fused";
const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 600000);

const extFromMime = (mime) => {
  const m = (mime || '').toLowerCase();
  if (m.includes('webm')) return 'webm';
  if (m.includes('wav')) return 'wav';
  if (m.includes('mpeg') || m.includes('mp3')) return 'mp3';
  if (m.includes('mp4')) return 'm4a';
  if (m.includes('ogg')) return 'ogg';
  return 'wav';
};

export const analyzeFused = async (audioData) => {
  const formData = new FormData();
  const hasName = audioData && typeof audioData.name === 'string' && audioData.name.length > 0;
  const fileName = hasName
    ? audioData.name
    : `recording.${extFromMime(audioData?.type)}`;
  formData.append("file", audioData, fileName);

  try {
    const response = await axios.post(API_URL, formData, {
      timeout: Number.isFinite(API_TIMEOUT_MS) ? API_TIMEOUT_MS : 600000,
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error calling fusion API:", error);
    const errorMessage = error.response ? JSON.stringify(error.response.data) : error.message;
    throw new Error(`Fusion API error: ${errorMessage}`);
  }
};
