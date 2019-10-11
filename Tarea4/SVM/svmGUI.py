"""
##################################################
    Instituto Tecnológico de Costa Rica
    Escuela de Ingeniería en Electrónica
    EL-5852 Intro. Reconocimiento de patrones
    Prof: Dr. Pablo Alvarado
    II Semestre 2019
    Tarea 4
    Estudiantes:  Esteban Agüero Pérez
                  Esteban Sanabria Villalobos
    Carnets: 2015097708
             2015070913
##################################################
"""

from sklearn import svm
from mnist import MNIST
from tkinter import *
from tkinter import messagebox

# Ruta de los datos de entrenamiento/prueba
mndata = MNIST('samples')

cantidadDatos = 5000

images, labels = mndata.load_training()
imagesTest, labelsTest = mndata.load_testing()

 # Se utiliza solo la cantidad establecida
sImages = images[:cantidadDatos]
sLabels = labels[:cantidadDatos]

# Instancia temporal
clf = None

b1 = "up"
xold, yold = None, None

# Dimensiones dibujo
xReal = 280 
yReal = 280
x = 28
y = 28
factorX = xReal//x 
factorY = yReal//y

# Genera la matriz del dibujo
def genMatriz(x,y):
	m = []
	for i in range(0, y):
		temp = []
		for j in range(0, x):
			temp.append(0)
		m.append(temp)
		temp = []
	return m

matrizDato = genMatriz(x,y)

def getMatriz(m):
	temp = []
	for fila in m:
		temp+=fila

	return temp

def borrar(widget):
	global matrizDato, x, y
	widget.delete("all")
	matrizDato = genMatriz(x,y)

def consultar(widget):
	global matrizDato, clf
	resultado = clf.predict([getMatriz(matrizDato)])
	messagebox.showinfo("Resultado","El posible número es "+str(resultado[0]))

def botonDown(event):
    global b1
    b1 = "down"

def botonUp(event):
    global b1, xold, yold
    b1 = "up"
    xold = None
    yold = None

def printMa(matri):
	for fila in matri:
		print(fila)
		print()

def motion(event):
    if b1 == "down":
        global xold, yold
        if xold is not None and yold is not None:
        	if event.x > 0 and event.x < xReal:
        		if event.y > 0 and event.y < yReal:
        			matrizDato[event.y//factorY][event.x//factorX]=255

        	event.widget.create_line(xold,yold,event.x,event.y,smooth=TRUE)
        xold = event.x
        yold = event.y

def main():
	global clf
	print("> Cantidad de datos de entrenamiento:", cantidadDatos)
	print("> Entrenando ...")
	clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.0001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
	clf.fit(sImages, sLabels)
	print("> Completado")

	ventana = Tk()
	ventana.title("Identificador de Números")
	areaDibujo = Canvas(ventana,width=xReal,height=yReal)
	areaDibujo.pack()
	areaDibujo.bind("<Motion>", motion)
	areaDibujo.bind("<ButtonPress-1>", botonDown)
	areaDibujo.bind("<ButtonRelease-1>", botonUp)
	button4=Button(ventana,fg="green",text="Consultar",command=lambda:consultar(areaDibujo))
	button4.pack(side=RIGHT)
	button4=Button(ventana,fg="red",text="Borrar",command=lambda:borrar(areaDibujo))
	button4.pack(side=LEFT)
	ventana.mainloop()
	return 0

main()