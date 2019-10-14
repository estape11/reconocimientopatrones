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

import keras_create_model as create_kModel
import keras_read_model as read_Kmodel

from tkinter import messagebox, Tk, Canvas, Button, RIGHT, LEFT, TRUE
import os.path
import numpy as np
from PIL import Image 

b1 = "up"
kModel = None
cantidadDatos = 5000
xold, yold = None, None

# Dimensiones dibujo
xReal = 280
yReal = 280
x = 28
y = 28
factorX = xReal//x
factorY = yReal//y

# Genera la matriz del dibujo


def genMatriz(x, y):
	m = []
	for _ in range(0, y):
		temp = []
		for _ in range(0, x):
			temp.append(0)
		m.append(temp)
		temp = []
	return m


matrizDato = genMatriz(x, y)


def getMatriz(m):
	temp = []
	for fila in m:
		temp += fila
	return temp


def borrar(widget):
	global matrizDato, x, y
	widget.delete("all")
	matrizDato = genMatriz(x, y)

def getClass(array):
  result = array[0]
  index = 0
  for item in result:
    if(item == 1.):
      break
    else:
      index += 1
  return index

def consultar(widget):
  global matrizDato, kModel
  prove = np.asmatrix(np.array(getMatriz(matrizDato)))
  resultado = kModel.predict(prove)
  messagebox.showinfo("Resultado", "El posible número es "+str(getClass(resultado)))

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
            matrizDato[event.y//factorY][event.x//factorX] = 255

        event.widget.create_line(xold, yold, event.x, event.y, smooth=TRUE)
      xold = event.x
      yold = event.y


def main():
  global kModel
  print("> Entrenando ...")
  if(kModel == None and not os.path.exists("model.json") and not os.path.exists("model.h5")): 
    kModel = create_kModel.create()
  else:
    kModel = read_Kmodel.read()
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
