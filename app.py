import streamlit as st
import torch
import torch.nn as nn
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

# ---------------------------
# MODEL: ECAPAClassifier
# ---------------------------
class ECAPAClassifier(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, num_classes=34):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# EMBEDDING LOADER (ECAPA)
# ---------------------------
spk_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

def ecapa_loader(path):
    waveform, sr = torchaudio.load(path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    embedding = spk_model.encode_batch(waveform).squeeze(1)
    return embedding.detach()

# ---------------------------
# SPEAKER LABELS
# ---------------------------
class_names = {
    0: "abdoulaye", 1: "abhishek", 2: "adam", 3: "adesh", 4: "agathe",
    5: "arisa", 6: "arthur", 7: "ayoub", 8: "camille", 9: "charlotte",
    10: "dilara", 11: "ekaterina", 12: "ella", 13: "hadi", 14: "henry",
    15: "himanshu", 16: "hugo", 17: "jermiah", 18: "jintian", 19: "joaquin",
    20: "kp", 21: "lou", 22: "marysheeba", 23: "matilde", 24: "maude",
    25: "piercarlo", 26: "saeed", 27: "sicheng", 28: "siwen", 29: "tim",
    30: "walid", 31: "yanlin", 32: "yiwei", 33: "yuying"
}

# ---------------------------
# LOAD YOUR TRAINED MODEL
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECAPAClassifier(num_classes=34).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Speaker Recognition", page_icon="🎙️")
st.title("🎙️ Speaker Recognition App")
st.markdown("Upload a `.wav` file and I’ll tell you who it sounds like.")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save to disk
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Play audio
    st.audio("temp.wav")

    # Predict
    with torch.no_grad():
        emb = ecapa_loader("temp.wav").unsqueeze(0).to(device)
        output = model(emb)

        if output.ndim == 3:
            output = output.squeeze(0).squeeze(0)
        elif output.ndim == 2:
            output = output.squeeze(0)

        probs = torch.softmax(output, dim=0)
        top_indices = probs.argsort(descending=True)[:3].tolist()

    st.subheader("🔮 Top Predictions:")
    for idx in top_indices:
        prob = torch.round(probs[idx] * 100).item()
        st.write(f"**{class_names[idx]}** — {prob}%")
