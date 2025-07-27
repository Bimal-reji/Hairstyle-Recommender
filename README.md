# 💇‍♂️ AI Haircut Recommender – Your Face, Your Fade!

Ever wondered *“what haircut suits my face shape?”* Let AI answer that for you.  
This project combines **computer vision**, **machine learning**, and **a touch of style** to recommend the perfect haircut — all you need is your face!

---
## 🚀 Features

- 📸 Upload a photo **or** use your **webcam**
- 🧠 Detects your **face shape** using OpenCV + MediaPipe
- 🧾 Classifies face shape with a dummy KNN model (Round, Oval, Square, Heart, Diamond)
- ✂️ Suggests **trendy hairstyles** based on your shape
- ✅ Visual feedback with **green facial landmarks**
- 🎨 Clean, dark-mode UI with **Tailwind CSS**
- 🌐 Powered by **Flask** (backend) and **Jinja2** (templating)

---

## 🛠 Tech Stack

| Tool        | Purpose                          |
|-------------|----------------------------------|
| Python      | Core logic + ML + image processing |
| OpenCV      | Face detection and annotation     |
| MediaPipe   | Facial landmark detection         |
| Flask       | Web backend                       |
| Jinja2      | Frontend templating               |
| Tailwind CSS| Beautiful modern UI               |
| KNN         | Dummy classifier for face shape   |

---

## 🔧 Setup Instructions

1. **Clone this repo**
   ```bash
   git clone https://github.com/yourusername/haircut-recommender.git
   cd haircut-recommender
