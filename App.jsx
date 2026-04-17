import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [topics, setTopics] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [uploadId, setUploadId] = useState("");

  const [numTopics, setNumTopics] = useState(5);
  const [numWords, setNumWords] = useState(10);
  const [passes, setPasses] = useState(10);
  const [autoTopics, setAutoTopics] = useState(false);
  const [minTopics, setMinTopics] = useState(2);
  const [maxTopics, setMaxTopics] = useState(10);

  const [autoTopicInfo, setAutoTopicInfo] = useState({
    enabled: false,
    best_num_topics: null,
    best_coherence_score: null,
    candidates: [],
  });

  const isLoading = isUploading || isProcessing;
  const canUpload = useMemo(() => !!file && !isLoading, [file, isLoading]);
  const canProcess = useMemo(() => !!uploadId && !isLoading, [uploadId, isLoading]);

  const topicStrengthData = useMemo(
    () =>
      topics.map((topic) => {
        const terms = topic.top_terms || [];
        const strength = terms.reduce((sum, term) => sum + (term.weight || 0), 0);
        return {
          topicLabel: `Topik ${topic.topic_id}`,
          topicId: topic.topic_id,
          strength: Number(strength.toFixed(4)),
        };
      }),
    [topics]
  );

  const coherenceData = useMemo(
    () =>
      (autoTopicInfo.candidates || []).map((item) => ({
        topicLabel: `${item.num_topics}`,
        coherenceScore: item.coherence_score,
      })),
    [autoTopicInfo]
  );

  const resetMessages = () => {
    setError("");
    setSuccess("");
  };

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setUploadId("");
    setTopics([]);
    setAutoTopicInfo({
      enabled: false,
      best_num_topics: null,
      best_coherence_score: null,
      candidates: [],
    });
    resetMessages();
  };

  const extractErrorMessage = (payload, fallback) => {
    if (typeof payload === "string" && payload.trim()) return payload;
    if (payload && typeof payload.detail === "string") return payload.detail;
    return fallback;
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Pilih file CSV terlebih dahulu.");
      return;
    }

    resetMessages();
    setTopics([]);
    setUploadId("");
    setAutoTopicInfo({
      enabled: false,
      best_num_topics: null,
      best_coherence_score: null,
      candidates: [],
    });
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(extractErrorMessage(data, "Gagal upload file."));
      }

      setUploadId(data.upload_id);
      setSuccess(`Upload berhasil (${data.num_documents} dokumen). Siap diproses.`);
    } catch (err) {
      setError(err.message || "Terjadi kesalahan saat upload file.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcess = async () => {
    if (!uploadId) {
      setError("Upload file dulu sebelum proses LDA.");
      return;
    }

    if (numWords < 1 || passes < 1) {
      setError("numWords dan passes harus minimal 1.");
      return;
    }

    if (!autoTopics && numTopics < 1) {
      setError("numTopics harus minimal 1.");
      return;
    }

    if (autoTopics && (minTopics < 2 || maxTopics < 2 || maxTopics < minTopics)) {
      setError("Untuk auto topics: minTopics/maxTopics minimal 2 dan maxTopics >= minTopics.");
      return;
    }

    resetMessages();
    setTopics([]);
    setAutoTopicInfo({
      enabled: false,
      best_num_topics: null,
      best_coherence_score: null,
      candidates: [],
    });
    setIsProcessing(true);

    try {
      const response = await fetch(`${API_BASE}/process`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          upload_id: uploadId,
          num_topics: numTopics,
          num_words: numWords,
          passes,
          auto_topics: autoTopics,
          min_topics: minTopics,
          max_topics: maxTopics,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(extractErrorMessage(data, "Gagal memproses LDA."));
      }

      setTopics(data.topics || []);
      setAutoTopicInfo(
        data.auto_topic_info || {
          enabled: false,
          best_num_topics: null,
          best_coherence_score: null,
          candidates: [],
        }
      );

      if (data.auto_topic_info?.enabled) {
        setSuccess(
          `Auto topic aktif. Jumlah topik terbaik: ${data.num_topics} (coherence: ${
            data.auto_topic_info.best_coherence_score
          }).`
        );
      } else {
        setSuccess(`Proses LDA selesai. ${data.topics?.length || 0} topik ditemukan.`);
      }
    } catch (err) {
      setError(err.message || "Terjadi kesalahan saat proses LDA.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 px-4 py-10">
      <div className="mx-auto w-full max-w-5xl rounded-2xl bg-white p-6 shadow-md md:p-8">
        <h1 className="text-2xl font-bold text-slate-800 md:text-3xl">LDA Topic Modeling</h1>

        <div className="mt-6 rounded-xl border border-slate-200 p-4">
          <h2 className="text-lg font-semibold text-slate-800">Upload File</h2>
          <div className="mt-3 flex flex-col gap-3 md:flex-row md:items-center">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full rounded-lg border border-slate-300 bg-slate-50 p-2 text-sm text-slate-700 file:mr-4 file:rounded-md file:border-0 file:bg-blue-600 file:px-4 file:py-2 file:text-white hover:file:bg-blue-700"
            />
            <button
              type="button"
              onClick={handleUpload}
              disabled={!canUpload}
              className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-400"
            >
              {isUploading ? "Uploading..." : "Upload"}
            </button>
          </div>
          {uploadId && (
            <p className="mt-3 text-xs text-slate-500">
              upload_id: <span className="font-mono">{uploadId}</span>
            </p>
          )}
        </div>

        <div className="mt-6 rounded-xl border border-slate-200 p-4">
          <h2 className="text-lg font-semibold text-slate-800">Konfigurasi & Proses LDA</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-3">
            <label className="text-sm text-slate-700">
              Top Words / Topik
              <input
                type="number"
                min={1}
                value={numWords}
                onChange={(e) => setNumWords(Math.max(1, Number(e.target.value) || 1))}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </label>
            <label className="text-sm text-slate-700">
              Passes
              <input
                type="number"
                min={1}
                value={passes}
                onChange={(e) => setPasses(Math.max(1, Number(e.target.value) || 1))}
                className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </label>
            <label className="inline-flex items-center gap-2 pt-7 text-sm font-medium text-slate-700">
              <input
                type="checkbox"
                checked={autoTopics}
                onChange={(e) => setAutoTopics(e.target.checked)}
                className="h-4 w-4 rounded border-slate-300"
              />
              Auto jumlah topik (coherence)
            </label>
          </div>

          {!autoTopics ? (
            <div className="mt-4 max-w-xs">
              <label className="text-sm text-slate-700">
                Jumlah Topik Manual
                <input
                  type="number"
                  min={1}
                  value={numTopics}
                  onChange={(e) => setNumTopics(Math.max(1, Number(e.target.value) || 1))}
                  className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
                />
              </label>
            </div>
          ) : (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <label className="text-sm text-slate-700">
                Minimum Topik
                <input
                  type="number"
                  min={2}
                  value={minTopics}
                  onChange={(e) => setMinTopics(Math.max(2, Number(e.target.value) || 2))}
                  className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
                />
              </label>
              <label className="text-sm text-slate-700">
                Maximum Topik
                <input
                  type="number"
                  min={2}
                  value={maxTopics}
                  onChange={(e) => setMaxTopics(Math.max(2, Number(e.target.value) || 2))}
                  className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
                />
              </label>
            </div>
          )}

          <button
            type="button"
            onClick={handleProcess}
            disabled={!canProcess}
            className="mt-4 rounded-lg bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {isProcessing ? "Memproses LDA..." : "Proses LDA"}
          </button>
        </div>

        {error && (
          <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        {success && (
          <div className="mt-4 rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
            {success}
          </div>
        )}

        {autoTopicInfo.enabled && coherenceData.length > 0 && (
          <section className="mt-8">
            <h2 className="text-lg font-semibold text-slate-800">Coherence Score per Jumlah Topik</h2>
            <div className="mt-4 h-72 w-full rounded-xl border border-slate-200 bg-slate-50 p-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={coherenceData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="topicLabel" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="coherenceScore" fill="#7c3aed" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}

        <section className="mt-8">
          <h2 className="text-lg font-semibold text-slate-800">Visualisasi Strength Topik</h2>
          {topicStrengthData.length > 0 && (
            <div className="mt-4 h-72 w-full rounded-xl border border-slate-200 bg-slate-50 p-3">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topicStrengthData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="topicLabel" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="strength" fill="#2563eb" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section className="mt-8">
          <h2 className="text-lg font-semibold text-slate-800">Hasil Topik</h2>

          {isProcessing && (
            <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-700">
              Sedang memproses model LDA, mohon tunggu...
            </div>
          )}

          {!isProcessing && topics.length === 0 && !error && (
            <p className="mt-3 text-sm text-slate-500">
              Belum ada hasil. Upload file, lalu klik tombol "Proses LDA".
            </p>
          )}

          {topics.length > 0 && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {topics.map((topic) => (
                <article
                  key={topic.topic_id}
                  className="rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-sm"
                >
                  <h3 className="text-base font-semibold text-slate-800">Topik #{topic.topic_id}</h3>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {(topic.top_terms || []).map((term) => (
                      <span
                        key={`${topic.topic_id}-${term.word}`}
                        className="rounded-full bg-blue-100 px-3 py-1 text-xs font-medium text-blue-700"
                        title={`weight: ${term.weight?.toFixed?.(4) ?? term.weight}`}
                      >
                        {term.word}
                      </span>
                    ))}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
