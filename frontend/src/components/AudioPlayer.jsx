import React from "react";

function AudioPlayer({ src }) {
  return (
    <audio controls src={src} style={{ width: "100%" }}>
      Your browser does not support the audio element.
    </audio>
  );
}

export default AudioPlayer;
