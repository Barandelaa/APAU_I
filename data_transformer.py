import re

def eliminar_lineas_con_exp_regular(archivo, expresion):
    # Compilar la expresión regular
    patron = re.compile(expresion)

    # Leer el contenido del archivo
    with open(archivo, 'r') as f:
        lineas = f.readlines()

    # Filtrar las líneas que no coinciden con la expresión
    lineas_filtradas = [linea for linea in lineas if not patron.search(linea)]

    # Escribir las líneas filtradas de nuevo en el archivo
    with open(archivo, 'w') as f:
        f.writelines(lineas_filtradas)

if __name__ == "__main__":
    nombre_archivo = 'prueba.data'  # Reemplaza con el nombre de tu archivo
    expresion_regular = r'ME2'  # Reemplaza con tu expresión regular

    eliminar_lineas_con_exp_regular(nombre_archivo, expresion_regular)
    print(f"Líneas eliminadas en {nombre_archivo} que coinciden con la expresión '{expresion_regular}'.")
