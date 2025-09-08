
<img width="864" height="743" alt="Screen Shot 2025-09-08 at 10 37 06" src="https://github.com/user-attachments/assets/4d749550-4272-4415-a417-42f3513f0e5d" />

# duDoanUngThu---randomForestdu1
Dự đoán Ung thư (Random Forest + Flask) - Final Fixed Version
=============================================================
RUN:
  pip install -r requirements.txt
  python model.py     # (tùy chọn) huấn luyện lại model.joblib
  python app.py       # chạy web

WHAT'S FIXED:
- Client-side validation prevents submitting empty or non-numeric fields.
- Server-side validation also checks and returns specific empty/invalid field names.
- Inputs preserve values after prediction or error.
- Invalid fields are highlighted and show messages.
- Model saved with joblib to avoid sklearn pickle version issues.
