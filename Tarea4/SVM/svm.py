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
from sklearn.metrics import confusion_matrix
from sklearn import svm
from mnist import MNIST
import random
from joblib import dump, load
import os.path

# Ruta de los datos de entrenamiento/prueba
mndata = MNIST('samples')

cantidadDatos = 5000 

images, labels = mndata.load_training()
imagesTest, labelsTest = mndata.load_testing()

 # Se utiliza solo la cantidad establecida
sImages = images[:cantidadDatos]
sLabels = labels[:cantidadDatos]
sImagesTest = imagesTest[:cantidadDatos]
sLabelsTest = labelsTest[:cantidadDatos]

filename = 'model.joblib'

def printNum(num):
	k = 0
	cadena = ""
	for i in range(0,28):
		for j in range(0,28):
			if num[k] != 0:
				cadena+="* "
			else:
				cadena+=str(num[k])+" "
			k+=1

		print(cadena)
		cadena=""

def main():
	if (os.path.exists(filename)):
		print("> Cargando los datos del modelo ...")
		# Carga los datos del modelo
		clf = load(filename)
		print("> Completado")

	else:
		print("> Cantidad de datos de entrenamiento:", cantidadDatos)
		print("> Entrenando ...")
		clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
		clf.fit(sImages, sLabels)
		print("> Guardando modelo ...")
		dump(clf, filename)
		print("> Completado")
 
	print("> Numero al azar")
	index = random.randrange(0, cantidadDatos)
	resultado = clf.predict([imagesTest[index]])

	# Imprime el numero de entrada
	print(mndata.display(imagesTest[index]),"\n")

	print("> Numero esperado:", labelsTest[index], "\n")

	print("> Numero detectado:", resultado[0],"\n")

	print("> Matriz de confusion")
	sLabelsTestPred =  clf.predict(sImages)
	print(confusion_matrix(list(sLabels), sLabelsTestPred,labels=[0,1,2,3,4,5,6,7,8,9]),"\n")

	return 0

main()