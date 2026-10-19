# Šta je API? — animacija (480p)

Animirano objašnjenje pojma **API** (Application Programming Interface): šta je, kako radi,
kako se koristi u kodu i gdje se danas primjenjuje.

- **Video:** [`api_animacija_480p.mp4`](api_animacija_480p.mp4) — 854×480, 30 fps, 2:10 min, H.264 + AAC (pozadinska muzika)
- **Izvor:** [`render.py`](render.py) — sve je nacrtano kodom (Pillow + NumPy), a ffmpeg ga spaja u video

## Sadržaj

| # | Vrijeme | Scena |
|---|---------|-------|
| 1 | 0:00 | Šta znači A-P-I |
| 2 | 0:10 | Analogija s restoranom (gost → konobar → kuhinja) |
| 3 | 0:31 | Kako API radi: zahtjev, obrada na serveru, JSON odgovor |
| 4 | 0:53 | Dijelovi API poziva: URL, HTTP metode, statusni kodovi |
| 5 | 1:14 | Primjer u Python kodu + koraci korištenja |
| 6 | 1:35 | Današnja primjena (plaćanja, mape, prijava, AI, IoT …) |
| 7 | 1:57 | Ukratko |

## Ponovno renderovanje

```bash
python3 render.py                 # napravi api_animacija_480p.mp4
python3 render.py --preview 5 40  # sačuvaj PNG sličice za zadane sekunde u preview/
```

Potrebni su Python 3, Pillow, NumPy, ffmpeg (libx264) i DejaVu fontovi.
