'use client';

import { useState, useEffect } from 'react';
import { api } from '@/lib/api';
import { isAxiosError } from 'axios';

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

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDocument[];
}

export default function Home() {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const loadDocuments = async () => {
    setLoadingDocs(true);
    try {
      const res = await api.get('/documents');
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
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const allowedExtensions = ['.pdf', '.txt', '.md', '.docx', '.html', '.htm', '.jpg', '.jpeg', '.png'];

    setUploading(true);
    setUploadMessage(null);

    let successCount = 0;
    let failCount = 0;
    const failedNames: string[] = [];

    for (const file of Array.from(files)) {
      const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();

      if (!allowedExtensions.includes(fileExt)) {
        failCount++;
        failedNames.push(file.name);
        continue;
      }

      const formData = new FormData();
      formData.append('file', file);

      try {
        await api.post('/upload', formData);
        successCount++;
      } catch (error: unknown) {
        console.error(error);
        failCount++;
        failedNames.push(file.name);
      }
    }

    await loadDocuments();
    e.target.value = '';

    if (failCount === 0) {
      setUploadMessage({ type: 'success', text: `${successCount} file(s) indexed successfully` });
    } else {
      setUploadMessage({
        type: 'error',
        text: `${successCount} succeeded, ${failCount} failed (${failedNames.join(', ')})`,
      });
    }

    setUploading(false);
    setTimeout(() => setUploadMessage(null), 4000);
  };

  const onDeleteDocument = async (filename: string) => {
    if (!confirm(`Delete "${filename}" permanently?`)) return;

    try {
      await api.delete('/documents', { data: { filename } });
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

    const userMessage: Message = { role: 'user', content: question };
    const historyForRequest = messages.map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const res = await api.post('/ask', { question, history: historyForRequest });
      const assistantMessage: Message = {
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.source_documents,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error: unknown) {
      console.error(error);
      const errorMsg = isAxiosError(error) ? error.response?.data?.detail || 'AI is not responding' : 'AI is not responding';
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${errorMsg}` }]);
    } finally {
      setLoading(false);
    }
  };

  const clearConversation = () => {
    if (!confirm('Clear the whole conversation?')) return;
    setMessages([]);
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
            <input type="file" multiple style={{ display: 'none' }} onChange={onUpload} disabled={uploading} />
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
          {messages.length === 0 && !loading && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center' }}>
              <div style={{ fontSize: 64, marginBottom: 20 }}>🔍</div>
              <div style={{ fontSize: 22, fontWeight: 600, color: '#1e293b' }}>Semantic Search</div>
              <div style={{ fontSize: 14, color: '#64748b', marginTop: 8, maxWidth: 400 }}>
                Upload documents (PDF, TXT, DOCX, images) and ask questions in natural language
              </div>
            </div>
          )}

          {messages.length > 0 && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
              <button onClick={clearConversation} style={{ fontSize: 11, color: '#94a3b8', background: 'none', border: '1px solid #e2e8f0', borderRadius: 8, padding: '4px 10px', cursor: 'pointer' }}>
                Clear conversation
              </button>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} style={{ marginBottom: 20 }}>
              {msg.role === 'user' ? (
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <div style={{ backgroundColor: '#3b82f6', color: 'white', padding: '12px 18px', borderRadius: 18, maxWidth: '70%', fontSize: 14 }}>
                    {msg.content}
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ backgroundColor: '#ffffff', padding: 20, borderRadius: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.06)', border: '1px solid #e2e8f0', fontSize: 14, lineHeight: 1.6, color: '#1e293b', whiteSpace: 'pre-wrap' }}>                    {msg.content}
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 8 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: '#64748b' }}>Sources ({msg.sources.length}) :</div>
                      {msg.sources.map((doc, i) => (
                        <div key={i} style={{ padding: 10, backgroundColor: '#ffffff', borderRadius: 10, border: '1px solid #e2e8f0', fontSize: 11, color: '#475569' }}>
                          <div style={{ fontWeight: 600, color: '#1e293b', marginBottom: 4 }}>
                            📄 {doc.filename || 'Document'} {doc.page ? `• Page ${doc.page}` : ''} {`• Résultat #${i + 1}`}
                          </div>
                          {doc.content && (
                            <div style={{ fontSize: 11, color: '#64748b', fontStyle: 'italic', marginTop: 4 }}>
                               {doc.content.length > 150 ? doc.content.substring(0, 150) + '...' : doc.content}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, color: '#64748b', fontSize: 14 }}>
              <span>⏳</span> Processing...
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