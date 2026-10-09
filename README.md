# هاشم — Hashem Agent 🎬

**استوديو تصميم شامل يعمل بالذكاء الاصطناعي، مجاني 100%، يعمل دون اتصال (مع نماذج محلية)، ويصدّر فيديو وصورًا وكتبًا بضغطة زر.**

A free, offline-first AI design studio for Windows that turns a plain-language
prompt into motion-graphics video, graphic design and fully-typeset books.

> ⚠️ صدقًا وبشفافية: لا يوجد برنامج "يفوق" نماذج الذكاء الرائدة. ما يميّز هاشم أنه
> **مجاني بالكامل، يعمل محليًا دون إنترنت، ويحوّل أي نموذج مجاني إلى استوديو إنتاج
> كامل** — وهذا ما لا تقدمه معظم الأدوات المدفوعة.

---

## ✨ ماذا يفعل؟

| المجال | القدرات |
|---|---|
| 🎬 **فيديو موشن جرافيك** | عناوين متحركة، نص عربي مشكّل صحيحًا، انتقالات (fade/slide/wipe/zoom/circle)، جسيمات، عدّادات، أشرطة تقدم، موسيقى خلفية مُولّدة، تعليق صوتي عربي عصبي، تصدير MP4/WebM/GIF حتى 4K |
| 🖼️ **تصميم صور** | بوستر، منشورات سوشيال (كل المنصات)، مصغّرات يوتيوب، شعارات، اقتباسات، لافتات، أغلفة كتب — بفلاتر وإزالة خلفية وعلامة مائية وQR |
| 📚 **تصميم كتب** | تحويل txt/md/docx إلى كتاب منسّق بفهرس وترقيم وأغلفة، وتصدير **PDF / EPUB / DOCX / HTML** |
| 🧠 **وكيل ذكي** | حلقة استدعاء أدوات، ذاكرة جلسات، تعلّم ذاتي يتحسن من تقييماتك |
| 🌐 **مجاني** | يعمل مع Ollama/LM Studio محليًا (بلا إنترنت وبلا مفاتيح)، أو Groq/OpenRouter/Gemini بمفاتيح مجانية |

## 🚀 التشغيل

```bash
pip install -r requirements.txt
python -m hashem            # يفتح الاستوديو في المتصفح
python -m hashem doctor     # فحص ذاتي
```

استوديو الويب يعمل على `http://127.0.0.1:8765` ويحتوي: محادثة، استوديو فيديو،
استوديو صور، استوديو كتب، معرض، وإعدادات + لوحة تعلّم.

### أمثلة في المحادثة
- «اصنع فيديو عن فوائد القهوة بثلاث نقاط»
- «صمم بوستر لمؤتمر التقنية بلوحة غروب»
- «حوّل هذا الملف إلى كتاب PDF بثيم داكن»

## 🪟 بناء ملف .exe لويندوز

البناء يتم تلقائيًا على GitHub Actions (windows-latest) عند الدفع، ويُرفع الناتج
كـ Artifact وإصدار Release. محليًا:

```bash
pip install -r requirements.txt -r build/requirements-windows.txt
python build/windows/build_exe.py
```

الناتج: `dist/HashemAgent.exe` — ملف واحد لا يحتاج تثبيت بايثون، ويضم ffmpeg
والخطوط العربية (Amiri, رخصة OFL) داخله.

## 🛠️ أدوات مجانية احترافية مستخدمة

FFmpeg (ترميز) • Pillow + NumPy (رسم) • arabic-reshaper + python-bidi (نص عربي) •
fpdf2 (PDF بنص حقيقي) • ebooklib (EPUB) • python-docx (Word) • edge-tts (نطق عصبي
مجاني) • qrcode • fonttools • jinja2/markdown • PyInstaller.

## 🧪 الاختبارات

```bash
python -m pytest -q
```

## 🏗️ البنية

`hashem/motion` محرك الفيديو • `hashem/design` استوديو الصور • `hashem/book` الكتب •
`hashem/audio` الموسيقى والنطق • `hashem/agent` الوكيل والأدوات • `hashem/learning`
التعلم الذاتي • `hashem/ui` الاستوديو • `hashem/llm` المزودات.
