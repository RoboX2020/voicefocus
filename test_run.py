import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torchaudio
from app import isolate_core, load_mono, load_models, TARGET_SR, _save_wav

d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test")
s1 = load_mono(os.path.join(d, "spk1_snt1.wav"))
s2 = load_mono(os.path.join(d, "spk2_snt1.wav"))
T = max(s1.shape[1], s2.shape[1])
s1 = torch.nn.functional.pad(s1, (0, T - s1.shape[1]))
s2 = torch.nn.functional.pad(s2, (0, T - s2.shape[1]))
mix = (s1 * 0.6 + s2 * 0.6)
mix_p = os.path.join(d, "mix.wav")
torchaudio.save(mix_p, mix, TARGET_SR)
print("mix seconds:", T / TARGET_SR)

ref_p = os.path.join(d, "spk1_snt2.wav")  # same speaker as s1, different sentence
out_t, out_o, status = isolate_core(mix_p, ref_p, 0, 0)
print("STATUS:", status)

# verify: isolated output should be closer to speaker 1 than speaker 2
_, spk = load_models()
e_out = spk.encode_batch(load_mono(out_t)).flatten()
e_s1 = spk.encode_batch(s1).flatten()
e_s2 = spk.encode_batch(s2).flatten()
cs = torch.nn.functional.cosine_similarity
print("sim(out, spk1) =", cs(e_out, e_s1, dim=0).item())
print("sim(out, spk2) =", cs(e_out, e_s2, dim=0).item())

# trim test
out_t2, out_o2, status2 = isolate_core(mix_p, ref_p, 0.5, 2.0)
w, fs = torchaudio.load(out_t2)
print("TRIM STATUS:", status2)
print("trimmed seconds:", w.shape[1] / fs)
