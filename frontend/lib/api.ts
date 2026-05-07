import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000', // L'adresse de ton FastAPI
});

export const askIA = async (question: string) => {
  try {
    const response = await api.post('/ask', { question });
    return response.data;
  } catch (error) {
    console.error("Erreur API:", error);
    throw error;
  }
};