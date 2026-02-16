import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000/predict";

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", selectedImage);

    try {
      const response = await axios.post(API_URL, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (response.data.status === "success") {
        setResult(response.data);
      } else if (response.data.status === "unrelated") {
        setResult(response.data); // Still set result to show the rejection UI
        setError(null);
      } else {
        setError(response.data.message || "An error occurred during analysis.");
      }
    } catch (err) {
      setError("Failed to connect to the backend server. Make sure it's running on port 8000.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white font-sans selection:bg-cyan-500/30">
      {/* Background Gradients */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-1/2 -left-24 w-80 h-80 bg-blue-600/10 rounded-full blur-3xl"></div>
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 py-12 md:py-20">
        {/* Header */}
        <header className="text-center mb-16 space-y-4">
          <div className="inline-flex items-center px-4 py-2 rounded-full border border-cyan-500/30 bg-cyan-500/5 text-cyan-400 text-sm font-medium mb-4 backdrop-blur-sm">
            <span className="relative flex h-2 w-2 mr-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            Real-time Aquatic Monitoring
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-500 to-indigo-600">
            AstraZite AI
          </h1>
          <p className="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto leading-relaxed">
            Revolutionizing microplastic detection using high-precision neural networks to safeguard our global water ecosystems.
          </p>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start">
          {/* Upload Card */}
          <div className="lg:col-span-5 space-y-6">
            <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 p-8 rounded-3xl shadow-2xl relative group transition-all duration-300 hover:border-cyan-500/30">
              <div className="space-y-6">
                <div
                  className={`relative aspect-square rounded-2xl border-2 border-dashed transition-all duration-300 flex flex-col items-center justify-center overflow-hidden
                    ${previewUrl ? 'border-cyan-500/50' : 'border-slate-700 hover:border-slate-500'}
                    ${!previewUrl && 'bg-slate-800/20'}
                  `}>
                  {previewUrl ? (
                    <>
                      <img src={previewUrl} alt="Preview" className="w-full h-full object-cover" />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <span className="text-sm font-medium">Click to change</span>
                      </div>
                    </>
                  ) : (
                    <div className="text-center p-6 space-y-3">
                      <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center mx-auto mb-2 text-3xl">
                        📸
                      </div>
                      <p className="text-slate-400 font-medium">Drop your sample image here</p>
                      <p className="text-slate-600 text-sm">Supports BMP, JPG or PNG</p>
                    </div>
                  )}
                  <input
                    type="file"
                    onChange={handleImageChange}
                    accept="image/*"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                </div>

                <button
                  onClick={handleUpload}
                  disabled={!selectedImage || loading}
                  className="w-full py-4 rounded-xl font-bold text-lg transition-all duration-300 transform
                    flex items-center justify-center gap-2
                    disabled:opacity-50 disabled:cursor-not-allowed
                    bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500
                    shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_rgba(6,182,212,0.5)] active:scale-95
                  "
                >
                  {loading ? (
                    <>
                      <svg className="animate-spin h-5 w-5 text-white" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analyzing...
                    </>
                  ) : (
                    "Initialize Detection Engine"
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-2xl flex items-start gap-3 animate-in fade-in slide-in-from-top-2">
                <span className="text-xl">⚠️</span>
                <p className="text-sm font-medium">{error}</p>
              </div>
            )}
          </div>

          {/* Results Analytics */}
          <div className="lg:col-span-7">
            {result?.status === "unrelated" ? (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-3xl p-10 text-center space-y-6 animate-in fade-in slide-in-from-right-4">
                <div className="w-20 h-20 bg-amber-500/20 rounded-full flex items-center justify-center mx-auto text-4xl text-amber-500">
                  🚫
                </div>
                <div className="space-y-2">
                  <h3 className="text-2xl font-bold text-amber-500">Unrelated Image Detected</h3>
                  <p className="text-slate-400 leading-relaxed max-w-md mx-auto">
                    {result.message}
                  </p>
                </div>
                <div className="pt-4 flex justify-center gap-4">
                  <div className="bg-slate-900/50 px-4 py-2 rounded-xl border border-slate-800 text-xs font-mono text-slate-500">
                    SALIENCY: {result.confidence}
                  </div>
                </div>
              </div>
            ) : result?.status === "success" ? (
              <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-700">
                <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-3xl p-8 overflow-hidden relative">
                  {/* AI Warning Banner */}
                  {result.image_type === 'AI Generated' && (
                    <div className="mb-6 bg-amber-500/10 border border-amber-500/30 p-4 rounded-2xl flex items-center justify-between animate-pulse">
                      <div className="flex items-center gap-3 text-amber-500">
                        <span className="text-2xl">✨</span>
                        <div>
                          <h4 className="font-bold text-sm uppercase tracking-tight">AI Generated Content Detected</h4>
                          <p className="text-[11px] text-amber-200/60 leading-tight">This sample appears to be synthetically generated. Results may not reflect real-world environmental data.</p>
                        </div>
                      </div>
                      <div className="text-[10px] font-bold font-mono bg-amber-500/20 px-2 py-1 rounded border border-amber-500/30 text-amber-500">
                        {result.image_type_confidence}
                      </div>
                    </div>
                  )}
                  {/* Result Header */}
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
                    <div>
                      <h3 className="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-1">Classification Status</h3>
                      <div className="flex items-center gap-3">
                        <span className={`text-3xl font-bold ${result.prediction === 'algae' ? 'text-green-400' : 'text-cyan-400'}`}>
                          {result.prediction.toUpperCase()}
                        </span>
                        <span className="px-3 py-1 bg-slate-800 rounded-lg text-xs font-bold text-slate-300 border border-slate-700">
                          {result.confidence} CONFIDENCE
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`px-4 py-2 rounded-xl text-sm font-bold border ${result.prediction === 'algae' ? 'bg-green-500/10 border-green-500/30 text-green-400' : 'bg-red-500/10 border-red-500/30 text-red-400'}`}>
                        {result.prediction === 'algae' ? 'NON-TOXIC SAMPLE' : 'MICROPLASTIC DETECTED'}
                      </span>
                    </div>
                  </div>

                  <p className="text-slate-300 leading-relaxed bg-slate-800/50 p-4 rounded-2xl border border-slate-700/50 mb-8 italic">
                    "{result.details}"
                  </p>

                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                    {Object.entries(result.counts).map(([type, count]) => (
                      <div key={type} className={`p-4 rounded-2xl border transition-all duration-300 ${count > 0 ? 'bg-cyan-500/10 border-cyan-500/30' : 'bg-slate-800/30 border-slate-700 text-slate-500'}`}>
                        <div className="text-xs font-bold uppercase tracking-widest mb-2 opacity-70">{type.replace(' I', '')}</div>
                        <div className={`text-2xl font-black ${count > 0 ? 'text-white' : 'text-slate-600'}`}>{count}</div>
                      </div>
                    ))}
                  </div>

                  {/* Purity Indicator */}
                  <div className="space-y-3">
                    <div className="flex justify-between items-end">
                      <span className="text-sm font-medium text-slate-400">Biological Sample Integrity</span>
                      <span className="text-2xl font-bold text-cyan-400">{result.algae_percentage}</span>
                    </div>
                    <div className="h-3 bg-slate-800 rounded-full overflow-hidden border border-slate-700">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-indigo-600 transition-all duration-1000 ease-out shadow-[0_0_15px_rgba(6,182,212,0.5)]"
                        style={{ width: result.algae_percentage }}
                      ></div>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-2xl p-6 text-center">
                    <div className="text-slate-500 text-xs font-bold uppercase mb-1">Total Particles</div>
                    <div className="text-3xl font-bold text-white">{result.total_particles}</div>
                  </div>
                  <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-2xl p-6 text-center">
                    <div className="text-slate-500 text-xs font-bold uppercase mb-1">Image Source</div>
                    <div className={`text-lg font-bold ${result.image_type === 'AI Generated' ? 'text-amber-400' : 'text-green-400'}`}>
                      {result.image_type}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono mt-1">{result.image_type_confidence} CONFIDENCE</div>
                  </div>
                  <div className="bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-2xl p-6 text-center">
                    <div className="text-slate-500 text-xs font-bold uppercase mb-1">Detection Logic</div>
                    <div className="text-sm font-bold text-indigo-400 uppercase tracking-tighter">ResNet18 Ensemble</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center p-12 border-2 border-dashed border-slate-800 rounded-3xl bg-slate-900/20">
                <div className="text-center space-y-4">
                  <div className="w-20 h-20 bg-slate-800/50 rounded-full flex items-center justify-center mx-auto text-4xl opacity-20">
                    🔬
                  </div>
                  <p className="text-slate-500 font-medium">Waiting for sample analysis data...</p>
                </div>
              </div>
            )}
          </div>
        </main>

        <footer className="mt-24 pt-8 border-t border-slate-900 text-center text-slate-600 text-sm">
          <p>© 2026 AstraZite Global Monitoring. Empowering environmental sustainability through Intelligence.</p>
        </footer>
      </div>
    </div>
  );
};

export default App;
