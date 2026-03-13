import React, { useState } from "react";

function NewYolo() {

  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    setResult(data);
  };

  return (
    <div style={{textAlign:"center"}}>

      <h1>Microplastic Detection</h1>

      <input
        type="file"
        onChange={(e)=>setFile(e.target.files[0])}
      />

      <br/><br/>

      <button onClick={handleUpload}>
        Analyze Image
      </button>

      {result && (

        <div>

          <h2>Detection Result</h2>

          <img
            src={`data:image/jpeg;base64,${result.image}`}
            width="500"
            alt="result"
          />

          <h3>Counts</h3>

          <p>Fiber : {result.counts.fiber}</p>
          <p>Film : {result.counts.film}</p>
          <p>Fragment : {result.counts.fragment}</p>
          <p>Pallet : {result.counts.pallet}</p>

          <h2>Total Microplastics : {result.total_microplastics}</h2>

        </div>

      )}

    </div>
  );
}

export default NewYolo;