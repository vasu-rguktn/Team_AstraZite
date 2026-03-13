import React, { useState, useEffect } from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

const PLASTIC_INFO = {
  fiber: {
    name: "Fiber",
    description: "Thread-like microplastics often originating from synthetic textiles (e.g., polyester, nylon) during laundry or industrial processes. They are among the most common microplastics found in water sources.",
    color: "#EF4444", // Red
    icon: "🧶"
  },
  film: {
    name: "Film",
    description: "Flat, thin pieces of plastic, typically from broken-down plastic bags, wrappers, or packaging materials. They have a high surface area and can easily transport chemical pollutants.",
    color: "#EAB308", // Yellow
    icon: "📄"
  },
  fragment: {
    name: "Fragment",
    description: "Irregularly shaped pieces from the breakdown of larger plastic objects like bottles, toys, or containers. These result from mechanical weathering and UV degradation over long periods.",
    color: "#A855F7", // Purple
    icon: "🧩"
  },
  pallet: {
    name: "Pellet (Nurdle)",
    description: "Small pre-production plastic granules (often called nurdles) used as raw material in plastic manufacturing. They are frequently lost during transport and are mistaken for food by marine life.",
    color: "#22C55E", // Green
    icon: "⚪"
  },
  foam: {
    name: "Foam",
    description: "Lightweight, porous fragments often from expanded polystyrene (Styrofoam) packaging or insulation. They fragment very easily and are notoriously difficult to remove from the environment.",
    color: "#3B82F6", // Blue
    icon: "🧼"
  },
  microbead: {
    name: "Microbead",
    description: "Perfectly spherical manufactured particles often used in exfoliating soaps, cosmetics, and industrial abrasives. Many countries have now banned these due to their immediate entry into water systems.",
    color: "#EC4899", // Pink
    icon: "🔮"
  }
};

const API_URL = "http://localhost:8000/predict";

const App = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showWelcome, setShowWelcome] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setShowWelcome(false), 3000);
    return () => clearTimeout(timer);
  }, []);

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
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.status === "success") {
        setResult(data);
      } else {
        setError(data.message || "An error occurred during analysis.");
      }
    } catch (err) {
      setError("Failed to connect to the analysis engine. Please ensure the backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getContaminationColor = (level) => {
    switch (level?.toLowerCase()) {
      case 'moderate': return 'text-yellow-400 border-yellow-400/30 bg-yellow-400/10';
      case 'high': return 'text-orange-500 border-orange-500/30 bg-orange-500/10';
      case 'extreme': return 'text-red-500 border-red-500/30 bg-red-500/10';
      default: return 'text-green-400 border-green-400/30 bg-green-400/10';
    }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 font-sans selection:bg-cyan-500/30 overflow-x-hidden">
      {/* Ambient Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-cyan-600/10 rounded-full blur-[120px] animate-pulse"></div>
        <div className="absolute top-[20%] -right-[5%] w-[30%] h-[30%] bg-indigo-600/10 rounded-full blur-[100px]"></div>
        <div className="absolute -bottom-[10%] left-[20%] w-[35%] h-[35%] bg-blue-600/10 rounded-full blur-[110px]"></div>
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 brightness-50 contrast-150"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-12 lg:py-16">
        {/* Navigation / Logo */}
        <nav className="flex justify-between items-center mb-16 animate-in fade-in slide-in-from-top-2 duration-1000">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <span className="text-white font-black text-xl italic">A</span>
            </div>
            <span className="text-2xl font-black tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              ASTRAZITE <span className="text-cyan-400 font-medium tracking-normal text-sm ml-1">AI</span>
            </span>
          </div>
        </nav>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">
          {/* Left Column: Input & Controls */}
          <div className="lg:col-span-4 space-y-8 animate-in fade-in slide-in-from-left-4 duration-700">
            <div className="space-y-4">
              <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tight leading-none">
                Detect <span className="text-cyan-400 italic">Microplastics</span> in Seconds.
              </h1>
              <p className="text-slate-400 text-lg leading-relaxed max-w-md">
                Empowering environmental research with precision YOLOv8 computer vision.
              </p>
            </div>

            <div className="p-1 rounded-3xl bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700/50 shadow-2xl relative group overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
              <div className="p-6 space-y-6 relative">
                <div
                  className={`relative aspect-square rounded-2xl border-2 border-dashed transition-all duration-500 flex flex-col items-center justify-center overflow-hidden
                                        ${previewUrl ? 'border-cyan-500/40 bg-slate-950/40' : 'border-slate-700 bg-slate-800/20 hover:border-slate-500'}
                                    `}>
                  {previewUrl ? (
                    <>
                      <img src={previewUrl} alt="Preview" className="w-full h-full object-cover transition-transform duration-700 group-hover:scale-105" />
                      <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center backdrop-blur-[2px]">
                        <div className="px-4 py-2 bg-white/10 rounded-full border border-white/20 text-xs font-bold uppercase tracking-widest">
                          Change Image
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-center p-8 space-y-4">
                      <div className="w-20 h-20 bg-slate-800/50 rounded-3xl flex items-center justify-center mx-auto mb-2 text-4xl shadow-inner border border-slate-700/50">
                        🔬
                      </div>
                      <div className="space-y-1">
                        <p className="text-white font-bold">Upload Microscopic Sample</p>
                        <p className="text-slate-500 text-sm">Drag and drop or click to browse</p>
                      </div>
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
                  className="w-full group relative py-4 rounded-2xl font-black text-lg transition-all duration-300 overflow-hidden
                                        disabled:opacity-40 disabled:cursor-not-allowed
                                        bg-white text-slate-950 hover:bg-cyan-400 active:scale-[0.98]
                                        shadow-[0_20px_40px_rgba(0,0,0,0.3)]
                                    "
                >
                  <span className="relative z-10 flex items-center justify-center gap-3">
                    {loading ? (
                      <>
                        <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        PROCESSING...
                      </>
                    ) : (
                      "RUN AI ANALYSIS"
                    )}
                  </span>
                </button>
              </div>
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-5 rounded-2xl flex items-start gap-4 animate-in fade-in slide-in-from-top-4">
                <span className="text-xl">🚨</span>
                <div className="space-y-1">
                  <p className="text-sm font-bold uppercase tracking-tight">System Error</p>
                  <p className="text-sm opacity-80">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-8 animate-in fade-in slide-in-from-right-8 duration-1000">
            {result ? (
              <div className="space-y-8">
                {/* Main Visual Segment */}
                <div className="grid grid-cols-1 xl:grid-cols-5 gap-8">
                  <div className="xl:col-span-3 space-y-6">
                    <div className="bg-slate-900/40 backdrop-blur-3xl border border-slate-700/50 rounded-[32px] overflow-hidden shadow-2xl relative group">
                      <div className="absolute top-4 left-4 z-20 flex gap-2">
                        <span className={`px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest border ${getContaminationColor(result.contamination_level)}`}>
                          {result.contamination_level} RISK
                        </span>
                        <span className="px-3 py-1 bg-slate-950/80 backdrop-blur-md rounded-full text-[10px] font-black uppercase tracking-widest border border-slate-700/50 text-cyan-400">
                          YOLOv8x ACTIVE
                        </span>
                      </div>
                      <div className="relative aspect-[4/3] bg-slate-950 flex items-center justify-center">
                        <img
                          src={`data:image/jpeg;base64,${result.image}`}
                          alt="Annotated"
                          className="w-full h-full object-contain"
                        />
                      </div>
                      <div className="p-6 bg-gradient-to-t from-slate-900 via-slate-900/90 to-slate-900/40">
                        <div className="flex justify-between items-end">
                          <div>
                            <h3 className="text-xs font-black text-slate-500 uppercase tracking-[0.2em] mb-1">Primary Detection</h3>
                            <p className="text-3xl font-black text-white italic uppercase tracking-tighter">
                              {result.prediction}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="text-xs font-black text-slate-500 uppercase tracking-[0.2em] mb-1">Total Particles</p>
                            <p className="text-3xl font-black text-cyan-400">
                              {result.total_microplastics}
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="xl:col-span-2 space-y-6">
                    {/* Classification Cards */}
                    <div className="grid grid-cols-1 gap-4">
                      {Object.entries(result.counts).map(([type, count]) => {
                        const info = PLASTIC_INFO[type.toLowerCase()] || { icon: "🪨", color: "#64748b" };
                        return (
                          <div key={type} className={`group p-5 rounded-2xl border transition-all duration-500 flex items-center justify-between ${count > 0 ? 'bg-slate-900/40 border-slate-700/50 hover:border-cyan-500/30' : 'bg-slate-900/20 border-slate-800 opacity-60'}`}>
                            <div className="flex items-center gap-4">
                              <div
                                className="w-12 h-12 rounded-xl flex items-center justify-center text-xl shadow-inner transition-all duration-300"
                                style={{ backgroundColor: count > 0 ? `${info.color}20` : '#1e293b', color: count > 0 ? info.color : '#475569' }}
                              >
                                {info.icon}
                              </div>
                              <div>
                                <p className={`text-sm font-black uppercase tracking-widest ${count > 0 ? 'text-white' : 'text-slate-500'}`}>{info.name || type}</p>
                                <p className="text-[10px] text-slate-500 uppercase font-bold tracking-tighter mt-1">Classification Match</p>
                              </div>
                            </div>
                            <div className={`text-2xl font-black ${count > 0 ? 'text-cyan-400' : 'text-slate-700'}`}>{count}</div>
                          </div>
                        );
                      })}
                    </div>

                    {/* AI vs Real Assessment */}
                    <div className="p-6 rounded-3xl bg-slate-900/40 border border-slate-800 flex flex-col justify-center gap-4 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 rounded-full blur-3xl"></div>
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Source Authenticity</span>
                        <span className="text-[10px] font-bold font-mono text-slate-500 bg-slate-950 px-2 py-1 rounded-md border border-slate-800">{result.image_type_confidence}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className={`text-lg font-black uppercase tracking-tighter ${result.image_type === 'AI Generated' ? 'text-amber-400 animate-pulse' : 'text-green-500'}`}>
                          {result.image_type}
                        </div>
                      </div>
                      <p className="text-[10px] text-slate-500 leading-tight italic">
                        {result.image_type === 'AI Generated'
                          ? "This sample shows characteristics of synthetic generation. Proceed with caution for empirical data."
                          : "Sample matches characteristics of real-world microscopic topography."}
                      </p>
                    </div>
                  </div>
                </div>

                {/* New Section: Charts and Descriptions */}
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-300">
                  {/* Distribution Chart */}
                  <div className="p-8 rounded-[32px] bg-slate-900/40 backdrop-blur-3xl border border-slate-700/50 shadow-2xl flex flex-col">
                    <div className="flex justify-between items-center mb-6">
                      <div>
                        <h3 className="text-xs font-black text-slate-500 uppercase tracking-[0.2em] mb-1">Visual Analytics</h3>
                        <p className="text-2xl font-black text-white italic uppercase tracking-tighter">Particle Distribution</p>
                      </div>
                      <div className="w-10 h-10 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400 border border-cyan-500/20">
                        📊
                      </div>
                    </div>

                    <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-4 h-[300px]">
                      {result.total_microplastics > 0 ? (
                        <>
                          <div className="h-full">
                            <ResponsiveContainer width="100%" height="100%">
                              <PieChart>
                                <Pie
                                  data={Object.entries(result.counts)
                                    .filter(([_, count]) => count > 0)
                                    .map(([type, count]) => ({
                                      name: PLASTIC_INFO[type.toLowerCase()]?.name || type,
                                      value: count,
                                    }))}
                                  cx="50%"
                                  cy="50%"
                                  innerRadius={40}
                                  outerRadius={70}
                                  paddingAngle={5}
                                  dataKey="value"
                                  stroke="none"
                                >
                                  {Object.entries(result.counts)
                                    .filter(([_, count]) => count > 0)
                                    .map(([type], index) => (
                                      <Cell key={`cell-${index}`} fill={PLASTIC_INFO[type.toLowerCase()]?.color || "#64748b"} />
                                    ))}
                                </Pie>
                                <Tooltip
                                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px', color: '#f8fafc' }}
                                  itemStyle={{ fontWeight: 'bold' }}
                                />
                                <Legend wrapperStyle={{ fontSize: '10px' }} />
                              </PieChart>
                            </ResponsiveContainer>
                          </div>

                          <div className="h-full border-l border-slate-800/50 pl-4">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart
                                layout="vertical"
                                data={Object.entries(result.counts)
                                  .filter(([_, count]) => count > 0)
                                  .map(([type, count]) => ({
                                    name: PLASTIC_INFO[type.toLowerCase()]?.name || type,
                                    count: count,
                                    color: PLASTIC_INFO[type.toLowerCase()]?.color || "#64748b"
                                  }))}
                                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                                <XAxis type="number" hide />
                                <YAxis
                                  dataKey="name"
                                  type="category"
                                  stroke="#64748b"
                                  fontSize={10}
                                  width={60}
                                  tick={{ fill: '#94a3b8', fontWeight: 'bold' }}
                                />
                                <Tooltip
                                  cursor={{ fill: 'transparent' }}
                                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '12px' }}
                                />
                                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                                  {Object.entries(result.counts)
                                    .filter(([_, count]) => count > 0)
                                    .map((entry, index) => (
                                      <Cell key={`cell-${index}`} fill={PLASTIC_INFO[entry[0].toLowerCase()]?.color || "#64748b"} />
                                    ))}
                                </Bar>
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </>
                      ) : (
                        <div className="col-span-2 h-full flex items-center justify-center text-slate-500 italic text-sm">
                          No particles detected to calculate analytics.
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Plastic Type Guide */}
                  <div className="p-8 rounded-[32px] bg-slate-900/40 backdrop-blur-3xl border border-slate-700/50 shadow-2xl space-y-6">
                    <div className="flex justify-between items-center">
                      <div>
                        <h3 className="text-xs font-black text-slate-500 uppercase tracking-[0.2em] mb-1">Environmental Lexicon</h3>
                        <p className="text-2xl font-black text-white italic uppercase tracking-tighter">Pollutant Dictionary</p>
                      </div>
                      <div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center text-indigo-400 border border-indigo-500/20">
                        📚
                      </div>
                    </div>

                    <div className="space-y-4 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
                      {Object.entries(PLASTIC_INFO).map(([key, info]) => (
                        <div key={key} className="p-4 rounded-2xl bg-slate-950/40 border border-slate-800/50 flex gap-4 hover:border-slate-600 transition-colors group/item">
                          <div
                            className="w-12 h-12 shrink-0 rounded-xl flex items-center justify-center text-2xl transition-all duration-300 group-hover/item:scale-110"
                            style={{ backgroundColor: `${info.color}15`, color: info.color }}
                          >
                            {info.icon}
                          </div>
                          <div className="space-y-1">
                            <p className="text-sm font-black text-white uppercase tracking-widest flex items-center gap-2">
                              {info.name}
                              {key === 'foam' || key === 'microbead' ? (
                                <span className="text-[8px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded uppercase font-bold tracking-tighter">Global Threat</span>
                              ) : null}
                            </p>
                            <p className="text-xs text-slate-400 leading-relaxed font-medium">
                              {info.description}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center p-16 border-2 border-dashed border-slate-800/50 rounded-[48px] bg-slate-900/10 transition-colors group">
                <div className="relative mb-8">
                  <div className="absolute inset-0 bg-cyan-500/20 rounded-full blur-[60px] animate-pulse"></div>
                  <div className="w-32 h-32 bg-slate-800/40 backdrop-blur-md rounded-[40px] flex items-center justify-center text-5xl shadow-2xl relative border border-slate-700/50 group-hover:rotate-6 transition-transform duration-500">
                    🌊
                  </div>
                </div>
                <div className="text-center space-y-3 max-w-sm">
                  <h3 className="text-2xl font-black text-slate-200 uppercase tracking-tighter">System Idle</h3>
                  <p className="text-slate-500 font-medium">
                    Please provide a high-resolution microscopic sample to begin the deep-layer object detection sequence.
                  </p>
                </div>
                <div className="mt-12 flex gap-3">
                  <div className="px-4 py-2 bg-slate-950/50 rounded-full border border-slate-800 text-[10px] font-bold text-slate-600 uppercase tracking-widest">ResNet18 Check</div>
                  <div className="px-4 py-2 bg-slate-950/50 rounded-full border border-slate-800 text-[10px] font-bold text-slate-600 uppercase tracking-widest transition-colors group-hover:border-cyan-500/30 group-hover:text-cyan-500/50">YOLO-V8 Engine</div>
                  <div className="px-4 py-2 bg-slate-950/50 rounded-full border border-slate-800 text-[10px] font-bold text-slate-600 uppercase tracking-widest">Auth-v1</div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer Metrics */}
        <footer className="mt-32 pt-12 border-t border-slate-900/50 flex flex-col md:flex-row justify-between items-center gap-8 animate-in fade-in slide-in-from-bottom-4 duration-1000">
          <div className="text-slate-500 text-sm font-medium">
            © 2026 AstraZite Global Monitoring. Built for Environmental Sustainability.
          </div>
          <div className="flex gap-8">
            <div className="text-center">
              <p className="text-[10px] font-black text-slate-600 uppercase tracking-[0.2em] mb-1">Latency</p>
              <p className="text-sm font-mono text-cyan-400/70">~420ms</p>
            </div>
            <div className="text-center">
              <p className="text-[10px] font-black text-slate-600 uppercase tracking-[0.2em] mb-1">Architecture</p>
              <p className="text-sm font-mono text-cyan-400/70">V8-Hybrid</p>
            </div>
            <div className="text-center">
              <p className="text-[10px] font-black text-slate-600 uppercase tracking-[0.2em] mb-1">Precision</p>
              <p className="text-sm font-mono text-cyan-400/70">±0.04</p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default App;

