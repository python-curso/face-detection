import cv2

# Xml de classificacao da OpenCV para faces
cascPath = "haarcascade_frontalface_default.xml"

# Setando a classificacao do xml da OpenCV
faceCascade = cv2.CascadeClassifier(cascPath)

# Ler a imagem
image = cv2.imread("3.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Fazer a deteccao da imagem
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30)
)

# Setar o retangulo amarelo
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 210), 2)

# Exibir a imagem
cv2.imshow("Titulo da imagem", image)
cv2.waitKey(0)
