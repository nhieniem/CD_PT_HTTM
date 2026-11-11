@echo off
echo ===== DANG CHAY THU NGHIEM HE THONG ADAS =====

:: 1. Thuc thi chuong trinh Python ADAS
python vehicledetection.py

:: 2. Tu dong kiem tra va tao commit sau khi tat video
echo.
echo ===== DANG LUU LICH SU TEST VAO GIT =====
git add .
git commit -m "test: Run ADAS system evaluation test"

:: 3. Day thang len GitHub
git push origin main

echo.
echo ===== HOAN THANH LUU LICH SU LE GITHUB! =====
pause
