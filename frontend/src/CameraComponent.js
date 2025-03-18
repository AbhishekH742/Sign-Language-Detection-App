import React, { useRef, useState, useEffect } from "react";
import axios from "axios";
import "./styles.css";

const CameraComponent = () => {
  const videoRef = useRef(null);
  const [prediction, setPrediction] = useState("Detecting...");
  const [lastSpoken, setLastSpoken] = useState(""); // Prevent repeating speech

  useEffect(() => {
    startCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("Error accessing camera:", error);
    }
  };

  const speakText = (text) => {
    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US"; // Adjust for different languages if needed
      utterance.rate = 1.0; // Speech speed
      speechSynthesis.speak(utterance);
    } else {
      console.warn("Speech synthesis is not supported in this browser.");
    }
  };

  const captureFrame = async () => {
    const video = videoRef.current;
    if (!video) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageData = canvas.toDataURL("image/jpeg");
    const base64Image = imageData.split(",")[1];

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", {
        image: base64Image,
      });

      const detectedSign = response.data.sign;
      setPrediction(detectedSign);

      // Speak only if a new sign is detected
      if (detectedSign !== lastSpoken) {
        speakText(detectedSign);
        setLastSpoken(detectedSign);
      }
    } catch (error) {
      console.error("Prediction error:", error);
      setPrediction("Error detecting");

      // Speak error message only if it's not already spoken
      if (lastSpoken !== "Error detecting") {
        speakText("Please show the sign properly.");
        setLastSpoken("Error detecting");
      }
    }
  };

  useEffect(() => {
    const interval = setInterval(captureFrame, 2000); // Capture frame every 2 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container">
      <h1 className="title">Real-Time Sign Language Detection</h1>
      <div className="video-container">
        <video ref={videoRef} autoPlay playsInline className="video" />
      </div>
      <p className="prediction">Detected Sign: <span>{prediction}</span></p>
    </div>
  );
};

export default CameraComponent;
