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

Se requieren de las siguientes bibliotecas para ejecutar los programas:
	>	sklearn
	>	numpy
	>	python-mnist
	>	tkinter

Para ejecutar el ejemplo de SVM
	>	$ cd SVM
	>	$ ./prepare.sh // Para poder emplear MNIST con SVM
	>	$ python3 svm.py // ejemplo de entrenamiento y matriz de confusión
		*	Si se desea re-entranar al modelo, se debe eliminar el archivo model.joblib
			>	$ rm model.joblib
			>	$ python3 svm.py

	>	$ python3 svmGUI.py // programa de entrada de gráfica

Para ejecutar el ejemplo de keras
	> $ cd keras

	"MAIN file, con interfaz de usurio, creación y lectura del modelo"

	> $python3 deepLearningKeras.py  

	"Archivo con función de creación y graficación de los errores"

	Para la creación de las graficas se debe descomentar la llamada en la ultima linea de codigo de este archivo.

	> $python3 keras_create_model.py

	"Archivo con lectura y reconstrucción del modelo"

	> $python3 keras_read_model.py
