import React from "react";
import AudioPlayer from "../components/AudioPlayer";

function SessionInspector() {
  return (
    <main style={{ padding: 24 }}>
      <h2>Session Inspector</h2>
      <AudioPlayer src="/sample_audio/dummy.wav" />
      <p>Transcript and analysis will appear here.</p>
    </main>
  );
}

export default SessionInspector;
