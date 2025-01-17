"""
gps.py

Matemática Discreta - IMAT
ICAI, Universidad Pontificia Comillas

Grupo: GP13B
Integrantes:
    - Eugenio Ribón Novoa
    - Ignacio Queipo de Llano Pérez-Gascón

Descripción:
Librería para la creación y análisis de las funciones requeridas en la navegación.
"""

from procesamiento_ficheros import process_data
from callejero import VELOCIDAD_CALLES_ESTANDAR, Cruce, Calle
from grafo import Grafo
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

path_cruces = "data/cruces.csv"
path_direcciones = "data/direcciones.csv"


def distancia(p1: tuple, p2: tuple) -> float:
    """Calcula distancia euclídea entre 2 puntos

    Args:
        p1 (tuple): primer punto
        p2 (tuple): segundi punti

    Returns:
        float: numero decimal con la distancia entre ambos puntos
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def distancia_cruces(cruce: Cruce, x: int, y: int) -> float:
    """Calcula la distancia entre un cruce y un punto de coordenadas x e y

    Args:
        cruce (Cruce): objeto tipo cruce
        x (int): coordenada x del puntp
        y (int): coordenada y del punto

    Returns:
        float: numero decimal con la distancia entre el cruce y el punto
    """
    return distancia((cruce.coord_x, cruce.coord_y), (x, y))


def calc_tiempo(distancia: float, velocidades: int) -> float:
    """Calcula el tiempo que se tardará en recorrer una distancia en base a la velocidad

    Args:
        distancia (float): distancia que se recorre
        velocidades (int): velocidad a la que se puede ir

    Returns:
        float: tiempo que se tarda
    """
    s = velocidades / 3.6
    m = distancia / 100

    return m / s


def calc_centros(cruces: pd.DataFrame, radio: int) -> pd.DataFrame:
    """Calcula los centros de los cruces cuyas coordenadas están a una distancia inferior a un radio, devolviendo
    el dataframe de cruces con las columans de las coordenadas x e y editadas para aquellos cruces cuyas distancias eran
    menores que el radio. Para la sustitución se usa la media entre los cruces contemplados

    Args:
        cruces (pd.DataFrame): data frame de cruces
        radio (int): distancia a tener en cuenta

    Returns:
        pd.DataFrame: data frame cruces modificado
    """

    coordenadas = cruces[
        [
            "Coordenada X (Guia Urbana) cm (cruce)",
            "Coordenada Y (Guia Urbana) cm (cruce)",
        ]
    ]
    df = cruces.copy()
    coord = set()
    for _, row in coordenadas.iterrows():
        x, y = row
        coord.add((x, y))

    coord = list(coord)
    cruces_cercanos = []
    for cruce in coord:
        gr = []
        for ncruce in coord:
            if distancia(cruce, ncruce) < radio:
                gr.append(ncruce)
                coord.remove(
                    ncruce
                )  # voy quitando para que no tenga siempre que comparar toda la lista
        if len(gr) > 1:
            cruces_cercanos.append(gr)

    dict_cruces = {}
    for gr in cruces_cercanos:
        mean_x = np.mean([c[0] for c in gr])
        mean_y = np.mean([c[1] for c in gr])
        dict_cruces[(mean_x, mean_y)] = gr

    for key in dict_cruces.keys():
        for coor in dict_cruces[key]:
            df.loc[
                (
                    (cruces["Coordenada X (Guia Urbana) cm (cruce)"] == coor[0])
                    & (cruces["Coordenada Y (Guia Urbana) cm (cruce)"] == coor[1])
                ),
                (
                    "Coordenada X (Guia Urbana) cm (cruce)",
                    "Coordenada Y (Guia Urbana) cm (cruce)",
                ),
            ] = (key[0], key[1])

    return df


def total_calles(df_cruces: pd.DataFrame, df_direcciones: pd.DataFrame) -> list:
    """Función que devuelve una lista con todos los códigos de vias únicos de los dataframes

    Args:
        df_cruces (pd.DataFrame)
        df_direcciones (pd.DataFrame)

    Returns:
        list: lista con todos los codigos de calles
    """
    df = pd.concat(
        [
            df_cruces["Codigo de vía tratado"],
            df_cruces["Codigo de via que cruza o enlaza"],
            df_direcciones["Codigo de via"],
        ]
    )
    return df.unique().tolist()


def crear_cruces(df_cruces: pd.DataFrame, calles: list[Calle]) -> list[Cruce]:
    """Genera todos los cruces del df devolviendo una lista con todos ellos. Además, asigna valores
    a todos los parámetros necesarios de los objetos cruce

    Args:
        df_cruces (pd.DataFrame): dataframe de cruces
        calles (list[Calle]): lista con todos los objetos calle generados

    Returns:
        list[Cruce]: lista con todos los objetos de tipo cruce
    """
    cruces_dict = {}
    dic_calles = {calle.codigo: calle for calle in calles}

    for _, row in df_cruces.iterrows():
        x = row["Coordenada X (Guia Urbana) cm (cruce)"]
        y = row["Coordenada Y (Guia Urbana) cm (cruce)"]
        coord = (x, y)

        if coord not in cruces_dict:
            cruces_dict[coord] = Cruce(x, y)
            cruces_dict[coord].agregar_cod_calle(
                row["Codigo de via que cruza o enlaza"]
            )
            cruces_dict[coord].agregar_cod_calle(row["Codigo de vía tratado"])
            cruces_dict[coord].agregar_dir_calle(row["Nombre de la via que cruza"])
            cruces_dict[coord].agregar_dir_calle(row["Nombre de la via tratado"])
            cruces_dict[coord].objetos_cruces.append(
                dic_calles[row["Codigo de via que cruza o enlaza"]]
            )

    return list(cruces_dict.values())


def crear_calles(df_cruces: pd.DataFrame, df_direcciones: pd.DataFrame) -> list[Calle]:
    """Genera todas las calles del df devolviendo una lista con todas ellas. Además, asigna valores
    a todos los parámetros necesarios de los objetos calle

    Args:
        df_cruces (pd.DataFrame): dataframe de cruces
        df_direcciones (pd.DataFrame): dataframe de dirreciones

    Returns:
        list[Calle]: lista con todos los objetos de tipo calle
    """
    cruces = df_cruces.copy()
    direcciones = df_direcciones.copy()

    calles = []
    cods_dirs = direcciones["Codigo de via"].unique()
    cods_calles = total_calles(cruces, direcciones)
    for cod_calle in cods_calles:
        calle = Calle()
        if cod_calle in cods_dirs:
            rows = direcciones[direcciones["Codigo de via"] == cod_calle]
            calle.codigo = cod_calle
            calle.tipo_calle = rows["Clase de la via"].iloc[0].strip()
            calle.nombre = rows["Nombre de la vía"].iloc[0].strip()

            for _, row in rows.iterrows():
                calle.coordx.append(row["Coordenada X (Guia Urbana) cm"])
                calle.coordy.append(row["Coordenada Y (Guia Urbana) cm"])
                calle.velocidad.add(calle.agg_velocidad())
                calle.literal_numeracion_calle.append(
                    row["Literal de numeracion"].strip()
                )

        else:
            rows = cruces[cruces["Codigo de vía tratado"] == cod_calle]
            if rows.empty:
                rows = cruces[cruces["Codigo de via que cruza o enlaza"] == cod_calle]
            calle.codigo = cod_calle
            calle.tipo_calle = rows["Clase de la via tratado"].iloc[0].strip()
            calle.nombre = rows["Nombre de la via tratado"].iloc[0].strip()

            for _, row in rows.iterrows():
                calle.coordx.append(row["Coordenada X (Guia Urbana) cm (cruce)"])
                calle.coordy.append(row["Coordenada Y (Guia Urbana) cm (cruce)"])
                calle.velocidad.add(VELOCIDAD_CALLES_ESTANDAR)

        calles.append(calle)

    return calles


def construir_grafo(cruces: list[Cruce], calles: list[Calle], tipo: str) -> Grafo:
    """Función que construye un grafo de nuestra clase Grafo en función del tipo deseado, pudiendo ser
    con un peso en las aristas de distancia o un peso de tiempo

    Args:
        cruces (list[Cruce]): lista con todos los cruces
        calles (list[Calle]): lista con todas las calles
        tipo (str): especifica qué tipo de grado se desea (distancia o tiempo)

    Returns:
        Grafo: grafo ya generado con todas las calles y todos los cruces
    """
    grafo = Grafo(dirigido=False)
    for cruce in cruces:
        grafo.agregar_vertice(cruce)
    for calle in calles:
        cruces_con_calle = [
            cruce for cruce in cruces if calle.codigo in cruce.codigo_calles_cruce
        ]
        calle.objetos_cruces = cruces_con_calle
        if calle.codigo is not None:
            x = calle.coordx
            y = calle.coordy

            ord_cruces = sorted(
                cruces_con_calle, key=lambda cruce: distancia_cruces(cruce, x[0], y[0])
            )
            for i in range(len(ord_cruces) - 1):
                p1 = (ord_cruces[i].coord_x, ord_cruces[i].coord_y)
                p2 = (ord_cruces[i + 1].coord_x, ord_cruces[i + 1].coord_y)
                dist = distancia(p1, p2)
                data = {
                    "nombre_calle": calle.nombre,
                    "coordx": calle.coordx,
                    "coordy": calle.coordy,
                    "literal_numeracion_calle": calle.literal_numeracion_calle,
                    "tipo_calle": calle.tipo_calle,
                }

                if tipo == "distancia":
                    grafo.agregar_arista(ord_cruces[i], ord_cruces[i + 1], data, dist)
                else:
                    grafo.agregar_arista(
                        ord_cruces[i],
                        ord_cruces[i + 1],
                        data,
                        calc_tiempo(dist, list(calle.velocidad)[0]),
                    )

    return grafo


def datos(path_cruces: str, path_direcciones: str, R=2000) -> tuple[pd.DataFrame]:
    """Función de procesamiento última, donde se devuelven los dataframes ya procesados y con las coordenadas x e y
    de los cruces cambiados en función de un radio especificado

    Args:
        path_cruces (str)
        path_direcciones (str)
        R (int, optional): Defaults to 2000.

    Returns:
        tuple[pd.DataFrame]: tupla con ambos dataframes
    """
    df_cruces, df_direcciones = process_data(path_cruces, path_direcciones)
    df_cruces.dropna(
        subset=[
            "Coordenada X (Guia Urbana) cm (cruce)",
            "Coordenada Y (Guia Urbana) cm (cruce)",
        ],
        inplace=True,
    )
    df_cruces = calc_centros(df_cruces, R)
    return df_cruces, df_direcciones


def crear_grafos(path_cruces: str, path_direcciones: str):
    """Función principal, procesa los datos y genera el grafo de distancias y de tiempos

    Args:
        path_cruces (str)
        path_direcciones (str)

    Returns:
        Grafo: grafo de distancias
        Grafo: grafo de tiempos
        Calle: lista con los objetos calle
        pd.DataFrame: df de cruces procesados y agrupados
        pd.DataFrame: df de direcciones procesadas
    """
    df_cruces_agrupados, df_direcciones = datos(path_cruces, path_direcciones)

    calles = crear_calles(df_cruces_agrupados, df_direcciones)

    cruces = crear_cruces(df_cruces_agrupados, calles)

    grafo_distancia = construir_grafo(cruces, calles, "distancia")
    grafo_tiempo = construir_grafo(cruces, calles, "tiempo")

    return grafo_distancia, grafo_tiempo, calles, df_cruces_agrupados, df_direcciones


def distance_to_coord(cruce: Cruce, coord: tuple) -> float:
    """
    Devuelve la distancia entre un cruce y una coordenada.

    Parameters:
    cruce (Cruce): El cruce.
    coord (tuple): La coordenada.

    Returns:
    float: La distancia entre un cruce y una coordenada.
    """
    return math.sqrt((cruce.coord_x - coord[0]) ** 2 + (cruce.coord_y - coord[1]) ** 2)


def calle_cerca_cruce(calle: Calle, literal_numeracion: str, inicio=False):
    """Devuelve el cruce más cercano a la calle y la instruccón para llegar al literal de
    numeración

    Args:
        calle (Calle): objeto calle
        literal_numeracion (str): literal de numeración de esa calle
        inicio (bool, optional): Defaults to False.

    Returns:
        Cruce: objeto de tipo cruce más cercano a la calle a la altura dada con el literal de numeración
        str: cadena con la instrucción para el iniio o el final de la ruta
    """
    coorx, coory = (
        calle.coordx[calle.literal_numeracion_calle.index(literal_numeracion)],
        calle.coordy[calle.literal_numeracion_calle.index(literal_numeracion)],
    )
    distancias = []

    for cruce in calle.objetos_cruces:
        distancia = distance_to_coord(cruce, (coorx, coory))
        distancias.append(distancia)
    indice = distancias.index(min(distancias))
    if inicio:
        return (
            calle.objetos_cruces[indice],
            f"Continua recto durante {int(min(distancias))} y posicionate en el cruce de {calle.objetos_cruces[indice].dirs_calles_cruce[0]} con {calle.objetos_cruces[indice].dirs_calles_cruce[1]}",
        )

    else:
        direc = direcciones_coordenadas(
            (
                calle.objetos_cruces[indice - 1].coord_x,
                calle.objetos_cruces[indice - 1].coord_y,
            ),
            (
                calle.objetos_cruces[indice].coord_x,
                calle.objetos_cruces[indice].coord_y,
            ),
            (coorx, coory),
        )
        if direc == 1:
            direc = "Gire a la derecha"
        elif direc == -1:
            direc = "Gire a la izquierda"
        else:
            direc = "Siga recto"

        return (
            calle.objetos_cruces[indice],
            f"{direc} y continue durante {int(min(distancias))} metros, finalmente, habrá llegado a su destino: {calle.nombre}",
        )


def ruta_minima(grafo: Grafo, origen: Cruce, destino: Cruce) -> list:
    """Función que devuelve la ruta minima dado un origen y un destino al que se quiere ir

    Args:
        grafo (Grafo): grafo de estudio
        origen (Cruce): nodo origen del grafo
        destino (Cruce): nodo destino del grafo

    Returns:
        list: lista con todos los cruces que se deben seguir para llegar al nodo destino
    """

    return grafo.camino_minimo(origen, destino)


def direcciones_nodos(nodo1: Cruce, nodo2: Cruce, nodo3: Cruce):
    """Funcion que. dados tres nodos, comprueba el ángulo que formas los 2 vectores que los componen y estudia si son
    positivos o negativos

    Args:
        nodo1 (Cruce)
        nodo2 (Cruce)
        nodo3 (Cruce)

    Returns:
        1 (int): dirección hacia la derecha
        -1 (int): dirección hacia la izquierda
        0 (int): no hay dirección
    """
    calle1 = set()

    for calle in nodo1.codigo_calles_cruce:
        calle1.add(calle)

    calle2 = set()

    for calle in nodo3.codigo_calles_cruce:
        calle2.add(calle)

    if calle1.intersection(calle2) != set():
        return 0

    vector1 = (nodo1.coord_x - nodo2.coord_x, nodo1.coord_y - nodo2.coord_y)
    vector2 = (nodo3.coord_x - nodo2.coord_x, nodo3.coord_y - nodo2.coord_y)

    producto_cruz = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    angulo_radianes = math.atan2(
        producto_cruz, vector1[0] * vector2[0] + vector1[1] * vector2[1]
    )
    angulo_grados = math.degrees(angulo_radianes)

    if angulo_grados > 0:
        resultado = 1
    elif angulo_grados < 0:
        resultado = -1
    else:
        resultado = 0

    return resultado


def direcciones_coordenadas(coord1: tuple, coord2: tuple, coord3: tuple) -> int:
    """Función igual a la anterior, simplemente que en vez de usar objetos del tipo cruce, usa tuplas

    Args:
        coord1 (tuple)
        coord2 (tuple)
        coord3 (tuple)
    Returns:
        1 (int): dirección hacia la derecha
        -1 (int): dirección hacia la izquierda
        0 (int): no hay dirección
    """

    v1 = (coord1[0] - coord2[0], coord1[1] - coord2[1])
    v2 = (coord3[0] - coord2[0], coord3[1] - coord2[1])

    producto_cruz = v1[0] * v2[1] - v1[1] * v2[0]

    angulo_radianes = math.atan2(producto_cruz, v1[0] * v2[0] + v1[1] * v2[1])
    angulo_grados = math.degrees(angulo_radianes)

    if angulo_grados > 0:
        resultado = 1
    elif angulo_grados < 0:
        resultado = -1
    else:
        resultado = 0

    return resultado


def instrucciones(ruta_minima: list, instruccion1: str, instruccionf: str) -> list:
    """Función que genera las intrucciones necesarias para seguir una ruta aplicando todas las funciones mencionadas
    a renglón anterior

    Args:
        ruta_minima (list): lista de cruces con la ruta minima
        instruccion1 (str): primera instruccion
        instruccionf (str): última instrucción

    Returns:
        list: Lista que contiene todas las instrucciones necesarias para seguir la ruta
    """
    nombre_calles = calles_ruta_minima(ruta_minima)
    instrucciones = [f"{instruccion1}"]
    metros_por_calle = 0

    for i in range(len(ruta_minima) - 1):
        if i == 0:
            metros_por_calle += distancia_cruces_m(ruta_minima[i], ruta_minima[i + 1])
        else:
            distancia_actual = distancia_cruces_m(ruta_minima[i], ruta_minima[i + 1])
            direccion = direcciones_nodos(
                ruta_minima[i - 1], ruta_minima[i], ruta_minima[i + 1]
            )

            if direccion == 0:
                metros_por_calle += distancia_actual
            else:
                if metros_por_calle != 0:
                    # instrucciones.pop() if len(instrucciones) > 1 else None
                    instrucciones.append(
                        f"Continúa {metros_por_calle} metros por {nombre_calles[i-1]}"
                    )
                    metros_por_calle = 0

                giro = "izquierda" if direccion == -1 else "derecha"
                instrucciones.append(f"Gira a la {giro} por {nombre_calles[i]}")
                instrucciones.append(
                    f"Continúa {distancia_actual} metros por la calle {nombre_calles[i]}"
                )

    if metros_por_calle > 0 and len(nombre_calles) > 1:
        instrucciones.pop()
        instrucciones.append(
            f"Continúa {metros_por_calle} metros por {nombre_calles[-2]}"
        )

    if nombre_calles:
        instrucciones.append(f"Llega a {nombre_calles[-1]}")

    instrucciones.append(instruccionf)

    return instrucciones


def distancia_cruces_m(cruce1: Cruce, cruce2: Cruce) -> int:
    """Función muy parecida a las de las distancias, simplemente que esta devuelve el resultado en metros

    Args:
        cruce1 (Cruce)
        cruce2 (Cruce)

    Returns:
        int: distancia en metros
    """
    diffx = cruce1.coord_x - cruce2.coord_x
    diffy = cruce1.coord_y - cruce2.coord_y

    qx = diffx**2
    qy = diffy**2

    suma_cuadrados = qx + qy
    dist = math.sqrt(suma_cuadrados)
    dist_en_metros = int(dist / 100)

    return dist_en_metros


def calles_ruta_minima(ruta_minima: list) -> list:
    """Función que dada una lista de objetos cruce con la ruta a seguir, devuelve el nombre de forma ordenada
    de todas las calles por las que se debe pasar para llegar al destino de la ruta

    Args:
        ruta_minima (list): lista de cruces con la ruta minima al destino

    Returns:
        list: lista de nombres de las calles
    """
    nombre_calles_ruta = [
        list(
            set(ruta_minima[i].dirs_calles_cruce).intersection(
                ruta_minima[i + 1].dirs_calles_cruce
            )
        )[0]
        for i in range(len(ruta_minima) - 1)
    ]

    return nombre_calles_ruta


def calle_to_obj(calle: str, lista_calles: list):
    """Función que dado el nombre de la calle y una lista con todos los objetos calle, devuelve el objeto correspondiente
    al nombre dado

    Args:
        calle (str): Nombre de la calle
        lista_calles (list): lista de los objetos calle

    Returns:
        Calle: objeto calle correspondiente
        str: literal de numeracion de esa calle
    """
    nombre = calle.split(",")[0]
    literal_de_numeracion = calle.split(",")[1].strip()
    for calle_obj in lista_calles:
        if (
            nombre == calle_obj.nombre
            and literal_de_numeracion in calle_obj.literal_numeracion_calle
        ):
            return calle_obj, literal_de_numeracion
    return None


def mostrar_instr(instrucciones: list) -> None:
    """Función para mostrar de forma mas legible la ruta a seguir por el ususario

    Args:
        instrucciones (list): lista de instrucciones
    """
    for instruccion in instrucciones:
        print(instruccion)
