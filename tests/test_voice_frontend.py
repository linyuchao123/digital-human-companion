import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path


HTML_PATH = Path(__file__).resolve().parents[1] / "integrated.html"


class VoiceFrontendTests(unittest.TestCase):
    def setUp(self):
        self.html = HTML_PATH.read_text(encoding="utf-8")

    def test_recording_is_bounded_and_releases_microphone(self):
        self.assertIn("setTimeout(()=>stopVoiceInput(),30000)", self.html)
        self.assertIn("getTracks().forEach(track=>track.stop())", self.html)
        self.assertIn("stopVoiceInput(false)", self.html)
        self.assertNotIn("stopMicOnly()", self.html)

    def test_server_recording_uses_pcm_wav_and_keeps_browser_fallback(self):
        self.assertIn("type:'audio',format:'wav'", self.html)
        self.assertIn("function _startBrowserVoiceInput()", self.html)
        self.assertIn("_isListening&&_voiceMode==='server'", self.html)

    @unittest.skipUnless(shutil.which("node"), "Node.js 未安装，跳过 WAV 编码运行测试")
    def test_interrupted_pending_tts_does_not_resume(self):
        function = re.search(r"async function speakText\(.*?(?=async function _loadTtsVoices)", self.html, re.S).group()
        script = """
        let ttsEnabled=true,_isListening=false,_voiceMode='',_ttsGeneration=0;
        let release,played=0;
        function _stopAudio(){_ttsGeneration++;}
        function _canUseServerTTS(){return new Promise(resolve=>release=resolve);}
        async function _speakViaServerTTS(){played++;return true;}
        function _speakViaBrowser(){played++;}
        """ + function + """
        (async()=>{const pending=speakText('旧回复');_stopAudio();release(true);
          await pending;console.log(played);})();
        """
        result = subprocess.run(["node"], input=script, text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "0")
        self.assertIn("if(generation!==_ttsGeneration||_audioSrc!==source) return", self.html)
        self.assertIn("if(!ttsEnabled) _stopAudio()", self.html)

    @unittest.skipUnless(shutil.which("node"), "Node.js 未安装，跳过 WAV 编码运行测试")
    def test_wav_encoder_resamples_to_16khz_mono_pcm(self):
        function = re.search(
            r"function _encodeVoiceWav\(.*?(?=function _startBrowserVoiceInput)",
            self.html,
            re.S,
        ).group()
        script = function + """
        (async()=>{
          const blob=_encodeVoiceWav([new Float32Array(48000).fill(0.25)],48000);
          const buffer=await blob.arrayBuffer(), v=new DataView(buffer);
          console.log(JSON.stringify({rate:v.getUint32(24,true),channels:v.getUint16(22,true),
            bits:v.getUint16(34,true),bytes:v.getUint32(40,true),sample:v.getInt16(44,true)}));
        })();
        """
        result = subprocess.run(["node"], input=script, text=True, capture_output=True, check=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload, {"rate": 16000, "channels": 1, "bits": 16, "bytes": 32000, "sample": 8191})


if __name__ == "__main__":
    unittest.main()
