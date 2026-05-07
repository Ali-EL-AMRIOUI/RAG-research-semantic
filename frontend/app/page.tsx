'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Document {
  name: string;
  chunks: number;
}

interface SourceDocument {
  content: string;
  filename?: string;
  page?: number;
  score?: number;
}

interface Answer {
  answer: string;
  source_documents?: SourceDocument[];
  processing_time_ms?: number;
}

export default function Home() {
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

  const loadDocuments = async () => {
    setLoadingDocs(true);
    try {
      const res = await axios.get(`${API_BASE_URL}/documents`);
      setDocuments(res.data.documents || []);
    } catch (error) {
      console.error('Error loading documents:', error);
    } finally {
      setLoadingDocs(false);
    }
  };

  useEffect(() => {
    loadDocuments();
  }, []);

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowedExtensions = ['.pdf', '.txt', '.md', '.docx', '.html', '.htm', '.jpg', '.jpeg', '.png'];
    const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!allowedExtensions.includes(fileExt)) {
      setUploadMessage({ type: 'error', text: 'Unsupported format. Accepted formats: PDF, TXT, MD, DOCX, HTML, JPG, PNG' });
      setTimeout(() => setUploadMessage(null), 3000);
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    
    setUploading(true);
    setUploadMessage(null);
    
    try {
      await axios.post(`${API_BASE_URL}/upload`, formData);
      setUploadMessage({ type: 'success', text: `"${file.name}" indexed successfully` });
      await loadDocuments();
      e.target.value = '';
    } catch (error: any) {
      console.error(error);
      const errorMsg = error.response?.data?.detail || 'Upload error';
      setUploadMessage({ type: 'error', text: `${errorMsg}` });
    } finally {
      setUploading(false);
      setTimeout(() => setUploadMessage(null), 3000);
    }
  };

  const onDeleteDocument = async (filename: string) => {
    if (!confirm(`Delete "${filename}" permanently?`)) return;
    
    try {
      await axios.delete(`${API_BASE_URL}/documents`, { data: { filename } });
      setUploadMessage({ type: 'success', text: `"${filename}" deleted` });
      await loadDocuments();
    } catch (error) {
      console.error(error);
      setUploadMessage({ type: 'error', text: `Deletion failed` });
    } finally {
      setTimeout(() => setUploadMessage(null), 3000);
    }
  };

  const onAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setAnswer(null);
    
    try {
      const res = await axios.post(`${API_BASE_URL}/ask`, { question });
      setAnswer(res.data);
      setQuestion('');  
    } catch (error: any) {
      console.error(error);
      alert(`Error: ${error.response?.data?.detail || 'AI is not responding'}`);
    } finally {
      setLoading(false);
    }
  };

  const truncateFilename = (name: string, maxLen: number = 25) => {
    if (name.length <= maxLen) return name;
    return name.substring(0, maxLen - 3) + '...';
  };

  return (
    <div style={{ display: 'flex', height: '100vh', backgroundColor: '#f0f4f8' }}>
      <aside style={{ width: 300, backgroundColor: '#1e2a3e', display: 'flex', flexDirection: 'column', boxShadow: '4px 0 12px rgba(0,0,0,0.1)' }}>
        <div style={{ padding: '28px 20px', borderBottom: '1px solid #334155' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 32 }}>🧠</span>
            <div>
              <div style={{ fontSize: 20, fontWeight: 'bold', color: '#ffffff' }}>Semantic RAG</div>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>Intelligent Search</div>
            </div>
          </div>
        </div>

        <div style={{ padding: '20px 20px', borderBottom: '1px solid #334155' }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8', marginBottom: 12, letterSpacing: '0.5px' }}>DOCUMENTS</div>
          <label style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, width: '100%', padding: '12px 16px', backgroundColor: '#3b82f6', border: 'none', borderRadius: 10, color: 'white', fontSize: 12, fontWeight: 500, cursor: 'pointer' }}>
            {uploading ? 'Indexing...' : 'Upload (PDF, TXT, DOCX, etc.)'}
            <input type="file" style={{ display: 'none' }} onChange={onUpload} disabled={uploading} />
          </label>
          {uploadMessage && (
            <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8, fontSize: 11, backgroundColor: uploadMessage.type === 'success' ? '#064e3b' : '#7f1d1d', color: uploadMessage.type === 'success' ? '#86efac' : '#fca5a5' }}>
              {uploadMessage.text}
            </div>
          )}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#94a3b8' }}>INDEXED ({documents.length})</div>
            <button onClick={loadDocuments} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 14 }}>⟳</button>
          </div>
          {loadingDocs ? (
            <div style={{ textAlign: 'center', padding: 16, color: '#64748b' }}>Loading...</div>
          ) : documents.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 16, color: '#64748b', fontSize: 12 }}>No documents</div>
          ) : (
            documents.map((doc, idx) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '10px 12px', backgroundColor: '#0f172a', borderRadius: 10, marginBottom: 8 }}>
                <span style={{ fontSize: 11, color: '#cbd5e1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }} title={doc.name}>
                  📄 {truncateFilename(doc.name)}
                </span>
                <button onClick={() => onDeleteDocument(doc.name)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', fontSize: 14 }}>🗑️</button>
              </div>
            ))
          )}
        </div>

        <div style={{ padding: '14px 20px', fontSize: 10, color: '#64748b', textAlign: 'center', borderTop: '1px solid #334155' }}>
          Qdrant • Groq • Next.js
        </div>
      </aside>

      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '32px 40px', overflow: 'hidden', backgroundColor: '#f0f4f8' }}>
        <div style={{ flex: 1, overflowY: 'auto', maxWidth: 900, margin: '0 auto', width: '100%' }}>
          {!answer && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center' }}>
              <div style={{ fontSize: 64, marginBottom: 20 }}>🔍</div>
              <div style={{ fontSize: 22, fontWeight: 600, color: '#1e293b' }}>Semantic Search</div>
              <div style={{ fontSize: 14, color: '#64748b', marginTop: 8, maxWidth: 400 }}>
                Upload documents (PDF, TXT, DOCX, images) and ask questions in natural language
              </div>
            </div>
          )}

          {loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
              <div style={{ fontSize: 48, marginBottom: 20 }}>⏳</div>
              <div style={{ fontSize: 16, color: '#64748b' }}>Processing...</div>
            </div>
          )}

          {answer && (
            <div>
              <div style={{ backgroundColor: '#ffffff', padding: 28, borderRadius: 20, boxShadow: '0 4px 12px rgba(0,0,0,0.08)', border: '1px solid #e2e8f0' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                  <span style={{ fontSize: 20 }}>💡</span>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#3b82f6', letterSpacing: '0.5px' }}>ANSWER</span>
                </div>
                <div style={{ fontSize: 15, lineHeight: 1.7, color: '#1e293b' }}>{answer.answer}</div>
                {answer.processing_time_ms && (
                  <div style={{ marginTop: 16, fontSize: 11, color: '#94a3b8', textAlign: 'right' }}>
                    ⚡ {answer.processing_time_ms} ms
                  </div>
                )}
              </div>

              {answer.source_documents && answer.source_documents.length > 0 && (
                <div style={{ marginTop: 32 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                    <span style={{ fontSize: 18 }}>📖</span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: '#475569' }}>SOURCES ({answer.source_documents.length})</span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {answer.source_documents.map((doc, i) => (
                      <div key={i} style={{ padding: 14, backgroundColor: '#ffffff', borderRadius: 16, border: '1px solid #e2e8f0', fontSize: 12, lineHeight: 1.5, color: '#334155', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 8, paddingBottom: 6, borderBottom: '1px solid #f1f5f9' }}>
                          <span style={{ fontWeight: 600, color: '#3b82f6', fontSize: 11 }}>#{i + 1}</span>
                          <span style={{ color: '#64748b', fontSize: 11 }}>📄 {doc.filename || 'Document'}</span>
                          {doc.page && doc.page > 0 && <span style={{ color: '#64748b', fontSize: 11 }}>p.{doc.page}</span>}
                          {doc.score && <span style={{ color: '#64748b', fontSize: 11 }}>{(doc.score * 100).toFixed(0)}%</span>}
                        </div>
                        <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', color: '#475569' }}>
                          {typeof doc === 'string' ? doc : doc.content || 'Content not available'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div style={{ maxWidth: 900, margin: '24px auto 0', width: '100%' }}>
          <form onSubmit={onAsk} style={{ position: 'relative' }}>
            <input
              type="text"
              style={{ width: '100%', padding: '16px 24px', paddingRight: 100, borderRadius: 48, border: '1px solid #cbd5e1', outline: 'none', fontSize: 14, backgroundColor: '#ffffff', boxShadow: '0 2px 8px rgba(0,0,0,0.05)' }}
              placeholder="Ask your question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={loading}
            />
            <button 
              type="submit" 
              style={{ position: 'absolute', right: 6, top: 6, padding: '10px 24px', backgroundColor: '#3b82f6', color: 'white', border: 'none', borderRadius: 40, cursor: 'pointer', fontSize: 13, fontWeight: 500 }}
              disabled={loading || !question.trim()}
            >
              Send
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}