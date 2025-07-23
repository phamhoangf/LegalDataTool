import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Health check
  healthCheck: () => api.get('/health'),

  // Topics
  getTopics: () => api.get('/topics'),
  createTopic: (data) => api.post('/topics', data),
  updateTopic: (id, data) => api.put(`/topics/${id}`, data),
  deleteTopic: (id) => api.delete(`/topics/${id}`),
  uploadDocument: (formData) => api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),

  // Data generation
  generateData: (data) => api.post('/generate', data),
  getGeneratedData: (topicId, dataType = null) => {
    const params = dataType ? { type: dataType } : {};
    return api.get(`/data/${topicId}`, { params });
  },

  // Labeling
  labelData: (data) => api.post('/label', data),

  // Export
  exportData: (dataType, topicId = null) => {
    const params = topicId ? { topic_id: topicId } : {};
    return api.get(`/export/${dataType}`, { 
      params,
      responseType: 'blob'
    });
  },

  // Statistics
  getStatistics: () => api.get('/stats'),
};

export default apiService;
