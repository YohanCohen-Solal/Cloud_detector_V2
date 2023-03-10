import './App.css';
//import React from 'react';
import Dropzone from './Dropzone';
import React, { useState, useEffect } from "react";
import api from "./api";

function App() {
  const [response, setResponse] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => {
    api
      .get("/")
      .then(res => {
        setResponse(res.data);
      })
      .catch(err => {
        console.log(err);
      });
    api.get('/result')
    .then(res => {
      setResult(res.data);
    })
    .catch(err => {
      console.log(err);
    });
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1 id="typo_ombre">Cloud Detector</h1>
        <p id="typo_ombre">Principle : Insert an image of cloud and we will tell you which type it is</p>
        <Dropzone />
        <h3>The type of cloud is : </h3>
        <p>{result}</p>
      </header>
    </div>
  );
}

export default App;
