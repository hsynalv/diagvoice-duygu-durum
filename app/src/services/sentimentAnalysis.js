import axios from 'axios';

const API_URL = "http://localhost:8000/analyze";

export const analyzeSentiment = async (text) => {
  try {
    const response = await axios.post(API_URL, { text });
    return response.data;
  } catch (error) {
    console.error("Error calling sentiment analysis API:", error);
    const errorMessage = error.response ? JSON.stringify(error.response.data) : error.message;
    throw new Error(`Sentiment analysis API error: ${errorMessage}`);
  }
};
