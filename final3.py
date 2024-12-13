import re
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import networkx as nx

class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.izquierda = None
        self.derecha = None

class DecisionNode:
    def __init__(self, expression, value=None):
        self.expression = expression  # Expresión a evaluar en el nodo
        self.value = value            # Valor lógico del nodo (True, False, o Nulo)
        self.children = []            # Hijos del nodo


class ASTNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
# Variable global para el contador de variables
contador_variables = 1
resultados_totales = []
estado_global = {}

proposiciones_existentes = {}  # Diccionario para llevar el control de las proposiciones

def extraer_proposiciones(enunciado, proposiciones_existentes):
    global contador_variables
    # Reemplazo inicial para los operadores lógicos
    enunciado_modificado = enunciado.replace(" y ", " ∧ ").replace(" o ", " ∨ ").replace(" NO ", " ~")
    
    # Separar las proposiciones usando el nuevo formato
    proposiciones = re.split(r'\s*∧\s*|\s*∨\s*', enunciado_modificado)
    variables = {}
    formula = ""
    operadores = re.findall(r'[∧∨]', enunciado_modificado)  # Obtener operadores en el mismo orden

    for i, p in enumerate(proposiciones):
        p = p.strip()
        if p == "":
            continue  # Ignorar entradas vacías

        # Verificar si la proposición está negada
        negada = p.startswith("~")  # Ver si hay un "~" al principio de la proposición

        # Eliminar el carácter de negación "~" de la proposición si está presente
        p_clean = p.replace("~", "").strip()

        # Verificar si la proposición ya existe en el conjunto de proposiciones
        if p_clean in proposiciones_existentes:
            # Si ya existe, usar el mismo identificador
            var = proposiciones_existentes[p_clean]
        else:
            # Si no existe en el conjunto de proposiciones, asignamos una nueva variable
            var = f"A{contador_variables}"
            contador_variables += 1
            # Guardar la proposición sin "NO"
            proposiciones_existentes[p_clean] = var

        # Crear la representación modificada de la proposición
        p_modificado = f"~{var}" if negada else var

        # Añadir la proposición al conjunto de variables
        variables[var] = p_clean

        if formula:  # Añadir el operador antes de la proposición excepto para la primera
            operador = operadores.pop(0) if operadores else ""  # Obtener el operador correspondiente
            formula += f" {operador} {p_modificado}"
        else:
            formula += p_modificado  # Añadir la primera proposición

    return variables, formula

def parse_expression(expression):
    # Analiza la expresión y detecta el operador principal
    if "∨" in expression:
        parts = expression.split("∨", 1)
        return parts[0].strip(), parts[1].strip(), "OR"
    elif "∧" in expression:
        parts = expression.split("∧", 1)
        return parts[0].strip(), parts[1].strip(), "AND"
    elif "~" in expression:
        return None, expression[1:].strip(), "NOT"
    else:
        return expression, None, None

def build_decision_tree(expression, parent=None):
    expr1, expr2, operator = parse_expression(expression)
    
    # Crear nodo actual
    node = DecisionNode(expression)
    
    if operator == "OR":
        # Crear hijos para OR (se evalúan ambos lados)
        child1 = build_decision_tree(expr1, node)
        child2 = build_decision_tree(expr2, node)
        node.children.extend([child1, child2])
        node.value = child1.value or child2.value  # Resultado OR
    elif operator == "AND":
        # Evaluar el primer término del AND
        child1 = build_decision_tree(expr1, node)
        # Evaluar el segundo término del AND
        child2 = build_decision_tree(expr2, node)

        # Procesar el valor del primer término
        value1 = not child1.value if "~" in child1.expression else child1.value

        # Crear un nodo para el primer término del AND
        if value1:
            node.children.append(child1)

            # Procesar el valor del segundo término solo si el primero es verdadero
            value2 = not child2.value if "~" in child2.expression else child2.value
            if value2:
                child1.children.append(child2)  # Añadir el segundo término como hijo del primero
            else:
                child1.value = False
        else:
            node.value = False

    elif operator == "NOT":
        # Invertir el valor lógico para NOT
        child = build_decision_tree(expr2, node)
        node.value = not child.value if child.value is not None else None
        node.children.append(child)
    else:
        # Nodo hoja (variable simple)
        # El valor es Nulo si "N" está en la expresión o True si no lo está
        node.value = "Nulo" if "N" in expression else True  
    return node

def plot_tree(node, x=0, y=0, x_offset=1, y_offset=-1, ax=None, parent_pos=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 2)
        ax.axis("off")
    
    # Dibujar el nodo actual
    node_label = f"{node.expression}\n({node.value})"
    ax.text(x, y, node_label, ha="center", va="center", bbox=dict(boxstyle="round", facecolor="lightblue"))
    
    # Dibujar la conexión al nodo padre
    if parent_pos:
        ax.plot([parent_pos[0], x], [parent_pos[1], y], 'k-')
    
    # Dibujar los hijos
    if node.children:
        n_children = len(node.children)
        for i, child in enumerate(node.children):
            child_x = x + x_offset * (-1 + 2 * i / (n_children - 1 if n_children > 1 else 1))
            child_y = y + y_offset
            plot_tree(child, x=child_x, y=child_y, x_offset=x_offset / 2, y_offset=y_offset, ax=ax, parent_pos=(x, y))
    
    # Mostrar la figura al final de la recursión
    if parent_pos is None:
        plt.show()

# Función de evaluación ajustada para devolver los resultados correctos
def evaluar_formula(formula, valores):
    # Reemplazar variables en la fórmula por sus valores
    for var, val in valores.items():
        # Asegurarse de manejar la negación correctamente
        if f"~{var}" in formula:
            formula = formula.replace(f"~{var}", str(not val))
        else:
            formula = formula.replace(var, str(val))
    
    # Evaluar la fórmula usando eval
    try:
        resultado = eval(formula.replace("∧", " and ").replace("∨", " or "))
    except:
        resultado = False
    return resultado

def tabla_de_verdad(variables, formula):
    lista_variables = list(variables.keys())
    combinaciones = list(itertools.product([False, True], repeat=len(lista_variables)))
    resultados = []

    for combinacion in combinaciones:
        valores = dict(zip(lista_variables, combinacion))
        resultado = evaluar_formula(formula, valores)
        valores['Resultado'] = resultado
        resultados.append(valores)

    return pd.DataFrame(resultados)

def inicializar_tabla_variables(formula, variables):
    estados = {}
    for var in set(re.findall(r'A\d+', formula)):
        # Asignar la proposición en lugar de true o false
        if f"~{var}" in formula:
            proposicion = f"NO {variables[var]}"  # Formato para negación
        else:
            proposicion = variables[var]  # Usar la proposición original

        # Asignar por defecto el valor booleano de NaN
        estados[var] = {
            "Proposicion": proposicion,
            "Valor Booleano": np.nan
        }

    # Retornar un DataFrame con el estado de las variables
    return pd.DataFrame.from_dict(estados, orient='index')


def actualizar_estado(resultados_totales):
    for resultado in resultados_totales:
        tabla_estados = pd.DataFrame.from_dict(resultado['tabla_estados'], orient='index', columns=["Proposicion", "Valor Booleano"])
        
        # Asegurarse de que la columna "Valor Booleano" es de tipo 'object' para aceptar bool
        tabla_estados["Valor Booleano"] = tabla_estados["Valor Booleano"].astype(object)
        
        while True:
            print("\nTabla de Estados de Variables Actual:")
            print(tabla_estados)

            variable = input("Ingresa la variable que deseas cambiar (o escribe 'fin' para terminar): ")
            if variable.lower() == 'fin':
                break

            if variable in tabla_estados.index:
                nuevo_valor = input(f"Ingresa el nuevo valor para {variable} (True/False): ").strip().capitalize()
                if nuevo_valor in ["True", "False"]:
                    tabla_estados.at[variable, "Valor Booleano"] = True if nuevo_valor == "True" else False
                else:
                    print("Valor no válido, ingresa 'True' o 'False'.")
            else:
                print("Variable no encontrada en la tabla de estados.")

        resultado['tabla_estados'] = tabla_estados.to_dict(orient='index')  # Actualizar el resultado con la tabla modificada


def build_ast(expression):
    operators = []
    operands = []

    def apply_operator():
        operator = operators.pop()
        right = operands.pop()
        if operator == "~":
            node = ASTNode(operator, right=right)
        else:
            left = operands.pop()
            node = ASTNode(operator, left=left, right=right)
        operands.append(node)

    i = 0
    while i < len(expression):
        if expression[i].isspace():
            i += 1
            continue
        elif expression[i] in "∧∨~":
            operators.append(expression[i])
            i += 1
        elif expression[i] == "(":
            operators.append(expression[i])
            i += 1
        elif expression[i] == ")":
            while operators and operators[-1] != "(":
                apply_operator()
            operators.pop()  # Remove "("
            i += 1
        else:
            j = i
            while j < len(expression) and (expression[j].isalnum() or expression[j] in "_"):
                j += 1
            operands.append(ASTNode(expression[i:j]))
            i = j

    while operators:
        apply_operator()

    return operands[0]

def generate_ast_graph_matplotlib(node):
    G = nx.DiGraph()
    def add_edges(node, parent=None):
        if node is not None:
            G.add_node(str(id(node)), label=node.value)
            if parent:
                G.add_edge(str(id(parent)), str(id(node)))
            add_edges(node.left, node)
            add_edges(node.right, node)
    add_edges(node)

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title('Árbol Sintáctico Abstracto')
    plt.show()


def generar_arbol_grafico(resultados_totales):
    for resultado in resultados_totales:
        formula = resultado['formula']
        ast_root = build_ast(formula)
        print(f"\nÁrbol Sintáctico de la fórmula: {formula}")
        generate_ast_graph_matplotlib(ast_root)

def combinar_formulas(formulas):
    # Usar paréntesis para agrupar cada fórmula individual
    formulas_agrupadas = [f"({formula})" for formula in formulas]
    # Combinar todas las fórmulas con el operador "∧"
    formula_combinada = " ∧ ".join(formulas_agrupadas)
    return formula_combinada

def combinar_y_generar_arbol(resultados_totales):
    formulas = [resultado['formula'] for resultado in resultados_totales]
    formula_combinada = combinar_formulas(formulas)
    print("\nFórmula combinada:", formula_combinada)
    ast_root = build_ast(formula_combinada)
    generate_ast_graph_matplotlib(ast_root)


def guardar_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def cargar_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def guardar_txt(resultados, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Cláusulas de Horn:\n")
        for resultado in resultados:
            variables = list(resultado['tabla_estados'].keys())
            estados = list(resultado['tabla_estados'].values())
            combinaciones = list(itertools.product(*[(var, f"¬{var}") if estado else (f"¬{var}", var) for var, estado in zip(variables, estados)]))
            
            for combinacion in combinaciones:
                clause = " ∧ ".join(combinacion)
                valor = "Verdadero" if any(estados) else "Falso"
                f.write(f"{clause} → {valor}\n")

def cargar_txt(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def mostrar_formulas():
    for resultado in resultados_totales:
        print("\nEnunciado:", resultado['enunciado'])
        print("Fórmula de lógica proposicional:", resultado['formula'])

def mostrar_tablas_verdad():
    for resultado in resultados_totales:
        tabla = pd.DataFrame(resultado['tabla'])
        print("\nTabla de Verdad:")
        print(tabla)
        
def mostrar_estado_variables(resultados_totales):
    for resultado in resultados_totales:
        # Asegúrate de que el DataFrame tiene las columnas correctas
        tabla_estados = pd.DataFrame.from_dict(resultado['tabla_estados'], orient='index', columns=["Proposicion", "Valor Booleano"])
        print("\nTabla de Estados de Variables:")
        print(tabla_estados)


def cambiar_valor_variables():
    for resultado in resultados_totales:
        tabla_estados = pd.DataFrame.from_dict(resultado['tabla_estados'], orient='index', columns=["Valor"])
        tabla_estados_actualizada = actualizar_estado(tabla_estados)
        resultado['tabla_estados'] = tabla_estados_actualizada

def mostrar_graficos():
    for resultado in resultados_totales:
        print(f"Enunciado: {resultado['enunciado']}")  # Muestra el enunciado correspondiente
        
        formula = resultado['formula']  # Obtiene la fórmula lógica
        # Generar el árbol de decisión a partir de la fórmula
        root = build_decision_tree(formula)  # Utiliza la función build_decision_tree

        # Graficar el árbol de decisión
        plot_tree(root)  # Utiliza la función plot_tree para dibujar el árbol

def obtener_variables_globales():
    variables_globales = {}
    for resultado in resultados_totales:
        for var in resultado['variables']:
            variables_globales[var] = resultado['tabla_estados'].get(var, {'Valor': False})  # Agrega el valor actual de la variable

    return variables_globales

def combinar_todas_formulas(resultados_totales):
    formulas = [resultado['formula'] for resultado in resultados_totales]
    formula_combinada = combinar_formulas(formulas)
    print("\nBase de Conocimiento:", formula_combinada)


def menu(resultados_totales):
    while True:
        print("\nMenú:")
        print("1. Mostrar fórmulas")
        print("2. Mostrar tablas de verdad")
        print("3. Mostrar estado de variables")
        print("4. Cambiar el valor de las variables")
        print("5. Mostrar gráficos")
        print("6. Generar árbol sintáctico")
        print("7. Arbol de decisiones")
        print("8. Salir")
        
        opcion = input("Selecciona una opción: ")

        if opcion == '1':
            mostrar_formulas()
        elif opcion == '2':
            mostrar_tablas_verdad()
        elif opcion == '3':
            mostrar_estado_variables(resultados_totales)
        elif opcion == '4':
            actualizar_estado(resultados_totales)
        elif opcion == '5':
            mostrar_graficos()
        elif opcion == '6':
            generar_arbol_grafico(resultados_totales)
        elif opcion == '7':
            combinar_y_generar_arbol(resultados_totales)
        elif opcion == '8':
            guardar_json(resultados_totales, 'resultados.json')
            guardar_txt(resultados_totales, 'clausulas_horn.txt')
            print("\nDatos guardados. Saliendo...")
            break
        else:
            print("Opción no válida, intenta nuevamente.")

# Inicio del programa
opcion = input("¿Quieres (1) introducir enunciados, (2) cargar resultados desde un archivo JSON o (3) cargar enunciados desde un archivo TXT? (1/2/3): ")

if opcion == '1':
    enunciados = []
    proposiciones_existentes = {}  # Conjunto para llevar el control de las proposiciones

    print("Introduce hasta 20 enunciados (escribe 'fin' para terminar):")
    while len(enunciados) < 20:
        enunciado = input(f"Enunciado {len(enunciados) + 1}: ")
        if enunciado.lower() == 'fin':
            break
        enunciados.append(enunciado)

    resultados_totales = []
    for enunciado in enunciados:
        variables, formula = extraer_proposiciones(enunciado, proposiciones_existentes)

        # Inicializar la tabla de estados con valores booleanos
        tabla_estados = inicializar_tabla_variables(formula, variables)

        # Guardar en resultados totales
        data_to_save = {
            'enunciado': enunciado,
            'variables': variables,
            'formula': formula,
            'tabla': tabla_de_verdad(variables, formula).to_dict(orient='records'),
            'tabla_estados': tabla_estados.to_dict(orient='index')
        }
        resultados_totales.append(data_to_save)

elif opcion == '2':
    filename = input("Introduce el nombre del archivo JSON a cargar: ")
    try:
        resultados_totales = cargar_json(filename)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print("Error al cargar el archivo:", e)

elif opcion == '3':
    filename = input("Introduce el nombre del archivo TXT a cargar: ")
    try:
        enunciados = cargar_txt(filename)
        resultados_totales = []
        proposiciones_existentes = {}
        for enunciado in enunciados:
            variables, formula = extraer_proposiciones(enunciado, proposiciones_existentes)

            # Inicializar la tabla de estados con valores booleanos
            tabla_estados = inicializar_tabla_variables(formula, variables)

            # Guardar en resultados totales
            data_to_save = {
                'enunciado': enunciado,
                'variables': variables,
                'formula': formula,
                'tabla': tabla_de_verdad(variables, formula).to_dict(orient='records'),
                'tabla_estados': tabla_estados.to_dict(orient='index')
            }
            resultados_totales.append(data_to_save)

    except FileNotFoundError:
        print("El archivo no fue encontrado. Asegúrate de que el nombre sea correcto.")

# Llamar al menú
menu(resultados_totales)

