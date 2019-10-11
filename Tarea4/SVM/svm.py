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
import random

# Ruta de los datos de entrenamiento/prueba
mndata = MNIST('samples')

cantidadDatos = 1000

images, labels = mndata.load_training()
imagesTest, labelsTest = mndata.load_testing()

 # Se utiliza solo la cantidad establecida
sImages = images[:cantidadDatos]
sLabels = labels[:cantidadDatos]
#sImagesTest = imagesTest[:cantidadDatos]
#sLabelsTest = labelsTest[:cantidadDatos]

def main():
	print("> Cantidad de datos de entrenamiento:", cantidadDatos)
	clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
	clf.fit(sImages, sLabels)

	index = random.randrange(0, cantidadDatos)
	resultado = clf.predict([imagesTest[index]])

	print(mndata.display(imagesTest[index]))

	print(len(imagesTest[index]))

	print("> Imagen")
	print(imagesTest[index])
	k = 0
	cadena = ""
	for i in range(0,28):
		for j in range(0,28):
			if imagesTest[index][k] != 0:
				cadena+="* "
			else:
				cadena+=str(imagesTest[index][k])+" "
			k+=1

		print(cadena)
		cadena=""

	print("> Label real")
	print(labelsTest[index])

	print("> Label calculado")
	print(resultado[0])
	return 0

main()