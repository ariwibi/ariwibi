import { useMemo, useState } from "react";

export default function App() {
  const [file, setFile] = useState(null);
  const [topics, setTopics] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [numTopics, setNumTopics] = useState(5);
  const [numWords, setNumWords] = useState(10);
  const [passes, setPasses] = useState(10);

  const canSubmit = useMemo(() => !!file && !isLoading, [file, isLoading]);

  const handleFileChange = (event) => {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    setTopics([]);
    setError("");
  };

  const handleProcess = async () => {
    if (!file) {
      setError("Silakan pilih file CSV terlebih dahulu.");
      return;
    }

    setIsLoading(true);
    setError("");
    setTopics([]);

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("num_topics", String(numTopics));
      formData.append("num_words", String(numWords));
      formData.append("passes", String(passes));

      const response = await fetch("http://localhost:8000/topic-modeling", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Gagal memproses LDA.");
      }

      setTopics(data.topics || []);
    } catch (err) {
      setError(err.message || "Terjadi kesalahan saat memproses data.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 py-10 px-4">
      <div className="mx-auto w-full max-w-4xl rounded-2xl bg-white p-6 shadow-md md:p-8">
        <h1 className="text-2xl font-bold text-slate-800 md:text-3xl">
          Topic Modeling LDA
        </h1>
        <p className="mt-2 text-sm text-slate-600">
          Upload file CSV, lalu klik tombol proses untuk menampilkan top words tiap topik.
        </p>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <div>
            <label className="mb-2 block text-sm font-medium text-slate-700">
              Jumlah Topik
            </label>
            <input
              type="number"
              min={1}
              value={numTopics}
              onChange={(e) => setNumTopics(Number(e.target.value) || 1)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-0 focus:border-blue-500"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-slate-700">
              Top Words / Topik
            </label>
            <input
              type="number"
              min={1}
              value={numWords}
              onChange={(e) => setNumWords(Number(e.target.value) || 1)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-0 focus:border-blue-500"
            />
          </div>

          <div>
            <label className="mb-2 block text-sm font-medium text-slate-700">
              Passes
            </label>
            <input
              type="number"
              min={1}
              value={passes}
              onChange={(e) => setPasses(Number(e.target.value) || 1)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm outline-none ring-0 focus:border-blue-500"
            />
          </div>
        </div>

        <div className="mt-6 space-y-4">
          <label className="block">
            <span className="mb-2 block text-sm font-medium text-slate-700">Upload CSV</span>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="block w-full rounded-lg border border-slate-300 bg-slate-50 p-2 text-sm text-slate-700 file:mr-4 file:rounded-md file:border-0 file:bg-blue-600 file:px-4 file:py-2 file:text-white hover:file:bg-blue-700"
            />
          </label>

          <button
            type="button"
            onClick={handleProcess}
            disabled={!canSubmit}
            className="inline-flex items-center justify-center rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {isLoading ? "Memproses LDA..." : "Proses LDA"}
          </button>
        </div>

        {error && (
          <div className="mt-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <section className="mt-8">
          <h2 className="text-lg font-semibold text-slate-800">Hasil Topik</h2>

          {isLoading && (
            <div className="mt-4 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-700">
              Sedang menjalankan pemodelan LDA, mohon tunggu...
            </div>
          )}

          {!isLoading && topics.length === 0 && !error && (
            <p className="mt-3 text-sm text-slate-500">
              Belum ada hasil. Upload file lalu klik "Proses LDA".
            </p>
          )}

          {topics.length > 0 && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {topics.map((topic) => (
                <article
                  key={topic.topic_id}
                  className="rounded-xl border border-slate-200 bg-slate-50 p-4 shadow-sm"
                >
                  <h3 className="text-base font-semibold text-slate-800">
                    Topik #{topic.topic_id}
                  </h3>
                  <div className="mt-3 flex flex-wrap gap-2">
                    {topic.top_words?.map((word) => (
                      <span
                        key={`${topic.topic_id}-${word}`}
                        className="rounded-full bg-blue-100 px-3 py-1 text-xs font-medium text-blue-700"
                      >
                        {word}
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
