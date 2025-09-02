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
  
  // Documents
  getDocuments: () => api.get('/documents'),
  createDocument: (data) => api.post('/documents', data),
  updateDocument: (id, data) => api.put(`/documents/${id}`, data),
  deleteDocument: (id) => api.delete(`/documents/${id}`),
  linkDocumentToTopic: (topicId, documentId, data = {}) => 
    api.post(`/topics/${topicId}/documents/${documentId}`, data),
  unlinkDocumentFromTopic: (topicId, documentId) => 
    api.delete(`/topics/${topicId}/documents/${documentId}`),
  
  // File upload
  uploadDocument: (formData) => api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  }),
  uploadDocumentFile: (formData) => api.post('/documents/upload', formData, {
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
  
  // Coverage Analysis
  analyzeCoverage: (topicId, settings) => api.post(`/topics/${topicId}/coverage`, settings),
  stopCoverageAnalysis: (topicId) => api.post(`/topics/${topicId}/coverage/stop`),
  analyzeBatchCoverage: (settings) => api.post('/coverage/batch', settings),

  // VanBan CSV Integration
  searchVanBanDocuments: (query = '', page = 1, limit = 50) => 
    api.get('/csv/search', { params: { q: query, limit, offset: (page - 1) * limit } }),
  previewVanBanDocument: (docId, full = false) => 
    api.get(`/csv/document/${docId}`, { params: { full: full ? 'true' : 'false' } }),
  importVanBanDocument: (docId) => api.post(`/csv/import/${docId}`, {}),
};

export default apiService;
