'''
Programadores:
Jose Pablo Mora Villalobos - B85326 - Participación: 50%
Juan José Valverde Campos  - B47200 - Participación: 50%
'''
from tracemalloc import stop
from turtle import right
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
# Fill with your code, or delete and import

def Gini(y):
    values = y.value_counts()
    quantity = y.size
    gini = 1
    for value in values: 
        gini -= (value/quantity)**2
    return gini

def Gini_split(ys):
    giniSplit = 0
    sonsQuantity = sum([son.size for son in ys])
    for son in ys:
        giniSplit += (Gini(son) * son.size / sonsQuantity)
    return giniSplit



class Node():

    def setLeaf(self, class_, count):
        '''
        Metodo para indicar que el nodo es hoja
        '''
        self.type = 'leaf'
        self.class_ = class_
        self.count = count



    def stopCondition(self,x, y, max_depth):
        '''
        Metodo recursivo para parar la creacion de nodos donde ademas se realiza la determinacion de como se
        considera debe de partirse el actual nodo
        '''
        if max_depth == 1:
            # Se detiene el nodo, por que llego a su maxima profundidad, por lo tanto el mismo se determina como hoja
            self.setLeaf(y.mode()[0], y.size)

        elif np.unique(np.array(y)).size == 1:
            # Se detiene el nodo, por que ya solo existe una unica clase para predecir, por lo tanto el mismo se determina como hoja
            self.setLeaf(y.mode()[0], y.size)

        else:
            # Se parte el nodo
            giniFather = Gini(pd.Series(y)) 
            giniMin = giniFather # Se considera el ginimin, al del mismo padre (se debe buscar hacerlo mas pequeno)
            # Atributos para determinar especificaciones del actual nodo
            columnName = None
            splitType = None
            divisionValue = None
            for column in x:
                # Se empieza a iterar por cada una de las columnas, donde primeramente se identifica si es un dato numerico o no
                if(x[column].dtype.name != "category" and x[column].dtype != "object" ):                                              
                    # Si entra aqui es un dato numérico   
                    minVal = x[column].min()
                    maxVal = x[column].max()
                    if (minVal != maxVal): # Si el valor maximo y minimo son iguales, no existe forma de separar datos numericos
                        intervals = np.linspace(minVal,maxVal,12)[1:-1]# Se realiza la separacion de 10 datos(sin incluir minimo y maximo)
                        for val in intervals: # Para cada uno de los intervalos se itera para ver si alguna separacion resulta adecuada
                            currentGini =  Gini_split([ pd.Series(y[x[column] <= val]),pd.Series(y[x[column] > val])])
                            if(currentGini < giniMin):  # Si el intervalo consigue un giniMin menor se queda
                                giniMin = currentGini
                                divisionValue = val
                                columnName = column
                                splitType = 'numerical'
                else:
                    # Si entra aqui es un dato categorico
                    categories = x[column].unique()
                    if categories.size != 1: # Si las categorias son iguales a 1 no se puede separar
                        for category in categories: # Se itera por las distintas categorias de la columna x
                            currentGini =  Gini_split([ pd.Series(y[x[column] == category]),pd.Series(y[x[column] != category])])
                            if(currentGini < giniMin):  # Si la categoria consigue un giniMin menor se queda
                                giniMin = currentGini
                                divisionValue = category
                                columnName = column
                                splitType = 'categorical'

            if (giniFather == giniMin): 
                # Dado que ninguna division por columna logró ninguna separacion se detiene aquí y se queda el nodo como hoja
                self.setLeaf(y.mode()[0], y.size)
            else:
                # Se determinan los atributos respectivos del nodo
                self.type = 'split'
                self.gini = giniFather
                self.count = y.size
                self.splitType = splitType
                self.splitColumn = columnName
                self.splitValue = divisionValue
                max_depth = max_depth if (max_depth == None) else (max_depth-1)
                # Se realiza el split del nodo para crear sus nuevos hijos
                if(splitType == 'categorical'): 
                    # Entra aqui si el split es de datos categoricos
                    self.childLeft = Node(x[x[columnName] == divisionValue]
                                            ,y[x[columnName] == divisionValue], max_depth)
                    self.childRight = Node(x[x[columnName] != divisionValue]
                                            ,y[x[columnName] != divisionValue], max_depth)
                else:
                    # Entra aqui si el split es de datos numericos
                    self.childLeft = Node(x[x[columnName] <= divisionValue]
                                            ,y[x[columnName] <= divisionValue], max_depth)
                    self.childRight = Node(x[x[columnName] > divisionValue]
                                        ,y[x[columnName] > divisionValue], max_depth)





    def toString(self):
        '''
        Metodo para poder conocer el nodod
        '''
        print(
        "\n-----------------------Nodo: ",
        "\ngini: ",self.gini,
        "\nclass_: ",self.class_,
        "\ncount: ",self.count,
        "\nvalues: ",self.values, 
        "\nsplitType: ",self.splitType, 
        "\nsplitValue: ",self.splitValue,
        "\nsplitColumn: ",self.splitColumn,
        "\ntype: ",self.type ,'\n')

    def printNodo(self):
        '''
        Metodo para imprimir el nodo segun un formato json
        '''
        if(self.type == 'leaf'):
            return {
            "type": "leaf",
            "class": self.class_,
            "count": self.count
            }
        else:
            return {
                "type": "split",
                "gini": self.gini,
                "count": self.count,
                "split-type": self.splitType,
                "split-column": self.splitColumn,
                "split-value": self.splitValue,
                "child-left": self.childLeft.printNodo(),
                "child-right": self.childRight.printNodo()
            }


    def __init__(self,x, y, max_depth):  
        '''
        Metodo de inicializacion de los nodos donde ademas se declaran las variables a utilizar
        '''

        self.gini = 0 
        self.class_ = None # Prediction in Node
        self.count = None # Number of samples
        self.values = [0,0] # Left Number of classes , right Number of other classes
        self.childLeft = None # Child node at left
        self.childRight = None # Child node at right
        self.splitType = None  # Column data type
        self.splitValue = None # Value to divide dataset
        self.splitColumn = None # Column to divide
        self.type = None    # Leaf or split
        self.stopCondition(x, y, max_depth)


class DecisionTree():
    '''
    Método constructor de la clase DecisionTree
    '''
    def __init__(self):
        self.root = None

    '''
    Crea la raíz del árbol de decisión y pasa los parátros variables de entrada y salida 
    'x' y 'y', y la profundidad del árbol. Con este método se realiza el entrenamiento 
    del árbol.
    '''
    def fit(self, x, y, max_depth = None):
        self.root = Node(x, y, max_depth)
    
    '''
    Este método se utiliza para hacer predicciones sobre un conjunto de datos. Se necesita
    haber llamado el método fit primero para poder hacer predicciones.
    '''
    def predict(self, x):
        res = []
        index_list = []
        if self.root == None:
            return  pd.Series([])
        else:
            for index,row in x.iterrows():
                node = self.root
                while node.type != 'leaf':
                    if (node.splitType == 'categorical'):
                        if (row[node.splitColumn] == node.splitValue):
                            node = node.childLeft
                        else:
                            node = node.childRight
                    else:
                        if (row[node.splitColumn] <= node.splitValue):
                            node = node.childLeft
                        else:
                            node = node.childRight
                res.append(node.class_)
                index_list.append(index)
            return pd.Series(res, index=index_list)
        
    '''
    Hace un llamado al método printNodo() de la clase nodo para generar un 
    diccionario con la información del árbol entrenado.
    '''
    def to_dict(self):
        if self.root == None:
            return {}
        else:
            return self.root.printNodo()

'''
Método para calcular los valores de la matriz de confusiones. Recibe los datos
predichos con el método predict de la clase DecisionTree y las clasificaciones
reales de los datos.
'''
def calculate_confusion_matrix(predict, real):
    confusion_matrix = [[0,0],[0,0]]
    for i in range(predict.size):
        row = 1 if predict.values[i] == 1 else 0
        col = 1 if real.values[i] == 1 else 0
        confusion_matrix[col][row]+=1
    return confusion_matrix
    # P =  Predict 
    # R =  Real 
    # [1 .(P=F,R=F) = ,2. (P=F,R=T)],[3. (P=T,R=F),4. (P=T,R=T)]
    # [4] [3]
    # [2] [1]






# Testing functions

def perform_test(name, function, inputs, outputs):
    print("Performing tests:",name)
    for i in range(len(inputs)):
        inval = inputs[i]
        val = function(inval)
        outval = outputs[i]
        if round(val,2) == outval:
            print("Case %i passed")
        else:
            print("Case %i failed")
            print("Input:",list(inval),"Output:",val,"Expected output:",outval)

def compare_dicts(dict1, dict2):
    if set(dict1.keys()) ^ set(dict2.keys()): print("Different keys");return False
    for key in dict2:
        if type(dict2[key])==dict:
            if not compare_dicts(dict1[key], dict2[key]): print("At key", key); return False
        elif type(dict2[key])==float:
            if dict1[key] - dict2[key] > 0.0001: print("At key", key);return False
        else:
            if dict1[key] != dict2[key]: print("At key", key, dict1[key],"!=",dict2[key]); return False
    return True

# Test cases for GINI

perform_test("GINI", Gini, [pd.Series(y) for y in [[0,0,0], [0,0,0,0,1,1,1]]], [0, 0.49])
perform_test("GINI_split", Gini_split, [[pd.Series(y) for y in [[0,0,0], [0,0,0,0,1,1,1]]]], [0.34])

# Test cases for Tree generation

tree = DecisionTree()
df = pd.read_csv('mushrooms.csv')
y = df['class']
x = df.drop(columns=['class'])
tree.fit(x,y)
mushrooms = tree.to_dict()

df = pd.read_csv('iris.data', names=['sepal-length','sepal-width','petal-length','petal-width','class'])
y = df['class']
x = df.drop(columns=['class'])
tree.fit(x,y)
iris = tree.to_dict()

df = pd.read_csv('titanic.csv')
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin']).dropna()

y = df['Survived']
x = df.drop(columns=['Survived'])
x['Pclass'] = x['Pclass'].astype('category')
tree.fit(x,y)
titanic = tree.to_dict()

import json
with open('lab_02_validation_results.json', 'r') as fin:
    data = json.load(fin)
if compare_dicts(mushrooms, data['mushrooms']):
    print("Mushrooms test case success!")
else:
    print("Mushrooms test case failed :(")
if compare_dicts(iris, data['iris']):
    print("Iris test case success!")
else:
    print("Iris test case failed :(")
if compare_dicts(titanic, data['titanic']):
    print("Titanic test case success!")
else:
    print("Titanic test case failed :(")

'''
Una de las posibles razones por las cuales el test de titanic falla puede ser debido a 
que para determinar las divisiones de cada nodo se utiliza un criterio distinto, específicamente
cuando una o más columnas da una métrica igual. En este caso el árbol de pruebas seleccionó
'Fare', mientras que el árbol generado con la clase decisionTree seleccionó 'Age'. En ambos casos
el gini obtenído fue de 3.5.
'''

# Métricas de desempeño
def calculate(total,tn,tp,fp,fn):
    trues = tn+ tp
    acc = trues / total
    prec = tp / (tp+fp)
    reca = tp / (tp+fn)
    f1 = 2 * prec*reca /(prec+reca)
    return acc,prec,reca,f1

def change(tn,tp,fp,fn):
    pivot = tn
    tn = tp
    tp = pivot
    pivot = fn
    fn = fp
    fp = pivot
    return tn,tp,fp,fn

def estadisticas(matrix):
    # P =  Predict 
    # R =  Real 
    # [1 .(P=F,R=F) = ,2. (P=F,R=T)],[3. (P=T,R=F),4. (P=T,R=T)]
    # [4=TP] [3=FP]
    # [2=FN] [1=TN]
    totalValues = sum(sum(x) for x in matrix)
    tn = matrix[0][0]
    tp = matrix[1][1] 
    fp = matrix[0][1]
    fn = matrix[1][0]
    accA,precA,recaA,f1A = calculate(totalValues,tn,tp,fp,fn)
    ## No aplica para otro caso
    sens = recaA
    spec = tn/tn+fp 
    tn,tp,fp,fn = change(tn,tp,fp,fn)
    accB,precB,recaB,f1B = calculate(totalValues,tn,tp,fp,fn)
    return {
                "Accuracy por clase a": accB,
                "Precision de clase a": precA,
                "Recall de clase a":recaA,
                "F1 de clase a": f1A,
                "Sensitividad": sens,
                "Specificidad":spec,
                "Accuracy por clase b": accB,
                "Precision de clase b": precB,
                "Recall de clase b": recaB,
                "F1 de clase b": f1B,
            }

def printF(estatistics):
    print("F1 de clase a: ", estatistics["F1 de clase a"])
    print("F1 de clase b: ", estatistics["F1 de clase b"],'\n')


#Test cases for class prediction

from sklearn.model_selection import PredefinedSplit, train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=21)

tree.fit(X_train, y_train, max_depth = 8)
predict = tree.predict(X_test)
if np.all(calculate_confusion_matrix(predict, y_test) == [[139,16],[27,79]]):
    print("Predict test case 1 success!")
else:
    print("Predict test case 1 failed")

values1 = calculate_confusion_matrix(predict, y_test)
print(values1)
printF(estadisticas(values1))

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=777)
tree.fit(X_train, y_train, max_depth = 8)
predict = tree.predict(X_test)
if np.all(calculate_confusion_matrix(predict, y_test) == [[145,19],[24,73]]):
    print("Predict test case 2 success!")
else:
    print("Predict test case 2 failed")

values2 = calculate_confusion_matrix(predict, y_test)
print(values2)
printF(estadisticas(values2))

# Modelo de sklearn
from sklearn.tree import DecisionTreeClassifier

features = ['Sex', 'Embarked','Pclass']
for col in features:
    df[col] = df[col].astype('category')   	# Indica que dicha variable es categórica
    mapping = df[col].cat.categories		# Almacena la codificación de las categorías
    df[col] = df[col].cat.codes			# Convierte los valores de la columna a la

y = df['Survived']
x = df.drop(columns=['Survived'])
x['Pclass'] = x['Pclass'].astype('category')


X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=21)

tree = DecisionTreeClassifier(criterion='gini', max_depth = 8)
tree.fit(X_train, y_train)
predict = pd.Series(tree.predict(X_test))
if np.all(calculate_confusion_matrix(predict, y_test) == [[145,19],[24,73]]):
    print("Predict test case 3 success!")
else:
    print("Predict test case 3 failed")

values3 = calculate_confusion_matrix(predict, y_test)
print(values3)
printF(estadisticas(values3))

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=777)

tree = DecisionTreeClassifier(criterion='gini', max_depth = 8)
tree.fit(X_train, y_train)
predict = pd.Series(tree.predict(X_test))
if np.all(calculate_confusion_matrix(predict, y_test) == [[145,19],[24,73]]):
    print("Predict test case 4 success!")
else:
    print("Predict test case 4 failed")

values4 = calculate_confusion_matrix(predict, y_test)
print(values4)
printF(estadisticas(values4))


'''
Los datos obtenidos a partir de las matrices anteriores muestran que para 
el test A con la semilla 21 para la separación de datos de prueba,
los valores de F1 tanto para la clase a como para la clase b son un poce 
mejores en el árbol de sklearn en comparación al generado con la clase 
decisionTree, 0.805 y 0.871 contra 0.786 y 0.866. Por otra parte, en el test B 
con una semilla de 777, la métrica de F1 para la categoría a fue ligeramente 
méjor en el árbol de sklear 0.775 contra un 0.77, pero en la categoría b el árbol
de la clase decisionTree tuvo un valor más alto 0.872 en comparación con 0.865.

En general, ambos árboles se comportaron relativemente igual para las pruebas con 
el set de datos del titanic, las diferencias entre ambos no son muy significativas.
También es interesante mencionar que ambos árboles tuvieron un mejor resultado de 
F1 para la clase b en comparación con la clase a. 
'''

'''
Una forma de mejorar el árbol es incluir algunas técnicas de podación.
Por ejemplo incluir un nivel de "impureza" el cuál define un umbral para determinar
si una división genera sufiente información para ser aplicada. En el estado actual
del proyecto, un nodo se divide si alguna columna genera un ginisplit menor que el gini 
actual. Con un nivel de 'impureza' se podría definir cuánto debería una cantidad mínima 
para considerar una división.
Por último, otra forma en la que se podría mejorar el árbol es incluyendo otrás métricas
para la separación de los nodos, por ejemplo al pasar la cantidad en que se quieren separar
los datos numéricos y no de la forma arbitaria de hacerlo por partes de a 10. O bien establecer
que dicha separación pueda hacerse para cada uno de los datos numéricos de formas distintas, 
pues piensese que la edad es menos separable que por ejemplo montos económicos.
'''

