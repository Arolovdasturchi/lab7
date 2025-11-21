import  cv2
import matplotlib.pyplot as plt

# Tasvirni yuklash
image = cv2.imread('ochi')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tasvirni threshold qilish (obyektni fonidan ajratish)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Konturlarni aniqlash
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Kontur bo'yicha bounding box chizish
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Natijani vizualizatsiya qilish
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
