"""
grafo.py

Matemática Discreta - IMAT
ICAI, Universidad Pontificia Comillas

Grupo: GP13B
Integrantes:
    - Eugenio Ribón Novoa
    - Ignacio Queipo de Llano Pérez-Gascón

Descripción:
Librería para la creación y análisis de grafos dirigidos y no dirigidos.
"""

from typing import List, Tuple, Dict
import networkx as nx
import sys
import numpy as np
import random
import heapq  # Librería para la creación de colas de prioridad

INFTY = sys.float_info.max  # Distincia "infinita" entre nodos de un grafo


class Grafo:
    """
    Clase que almacena un grafo dirigido o no dirigido y proporciona herramientas
    para su análisis.
    """

    def __init__(self, dirigido: bool = False):
        """Crea un grafo dirigido o no dirigido.

        Args:
            dirigido (bool): Flag que indica si el grafo es dirigido (verdadero) o no (falso).

        Returns:
            Grafo o grafo dirigido (según lo indicado por el flag)
            inicializado sin vértices ni aristas.
        Eugenio
        """

        # Flag que indica si el grafo es dirigido o no.
        self.dirigido = dirigido

        """
        Diccionario que almacena la lista de adyacencia del grafo.
        adyacencia[u]:  diccionario cuyas claves son la adyacencia de u
        adyacencia[u][v]:   Contenido de la arista (u,v), es decir, par (a,w) formado
                            por el objeto almacenado en la arista "a" (object) y su peso "w" (float).
        """
        self.adyacencia: Dict[object, Dict[object, Tuple[object, float]]] = {}

    #### Operaciones básicas del TAD ####
    def es_dirigido(self) -> bool:
        """Indica si el grafo es dirigido o no

        Args: None
        Returns: True si el grafo es dirigido, False si no.
        Raises: None
        """
        if self.dirigido:
            return True
        else:
            return False

    def agregar_vertice(self, v: object) -> None:
        """Agrega el vértice v al grafo.

        Args:
            v (object): vértice que se quiere agregar. Debe ser "hashable".
        Returns: None
        Raises:
            TypeError: Si el objeto no es "hashable".
        """

        try:
            self.adyacencia[v] = {}

        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def agregar_arista(
        self, s: object, t: object, data: object = None, weight: float = 1
    ) -> None:
        """Si los objetos s y t son vértices del grafo, agrega
        una arista al grafo que va desde el vértice s hasta el vértice t
        y le asocia los datos "data" y el peso weight.
        En caso contrario, no hace nada.

        Args:
            s (object): vértice de origen (source).
            t (object): vértice de destino (target).
            data (object, opcional): datos de la arista. Por defecto, None.
            weight (float, opcional): peso de la arista. Por defecto, 1.
        Returns: None
        Raises:
            TypeError: Si s o t no son "hashable".
        """
        try:
            if s and t in self.adyacencia.keys():
                if self.dirigido:
                    self.adyacencia[s][t] = (data, weight)
                else:
                    self.adyacencia[s][t] = (data, weight)
                    self.adyacencia[t][s] = (data, weight)
            else:
                print(f"Los vertices dados no existen")

        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def eliminar_vertice(self, v: object) -> None:
        """Si el objeto v es un vértice del grafo lo elimina.
        Si no, no hace nada.

        Args:
            v (object): vértice que se quiere eliminar.
        Returns: None
        Raises:
            TypeError: Si v no es "hashable".
        """
        try:
            if v in self.adyacencia.keys():
                del self.adyacencia[v]

            for clave in self.adyacencia.keys():
                self.adyacencia[clave].pop(v, None)

        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def eliminar_arista(self, s: object, t: object) -> None:
        """Si los objetos s y t son vértices del grafo y existe
        una arista de s a t la elimina.
        Si no, no hace nada.

        Args:
            s: vértice de origen de la arista (source).
            t: vértice de destino de la arista (target).
        Returns: None
        Raises:
            TypeError: Si s o t no son "hashable".
        """
        try:
            if t in self.adyacencia[s].keys():
                del self.adyacencia[s][t]
        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def obtener_arista(self, s: object, t: object) -> Tuple[object, float] or None:
        """Si los objetos s y t son vértices del grafo y existe
        una arista de s a t, devuelve sus datos y su peso en una tupla.
        Si no, devuelve None

        Args:
            s: vértice de origen de la arista (source).
            t: vértice de destino de la arista (target).
        Returns:
            Tuple[object,float]: Una tupla (a,w) con los datos "a" de la arista (s,t) y su peso
                "w" si la arista existe.
            None: Si la arista (s,t) no existe en el grafo.
        Raises:
            TypeError: Si s o t no son "hashable".
        """
        try:
            if t in self.adyacencia[s].keys():
                return self.adyacencia[s][t]
            else:
                return None
        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def lista_vertices(self) -> List[object]:
        """Devuelve una lista con los vértices del grafo.

        Args: None
        Returns:
            List[object]: Una lista [v1,v2,...,vn] de los vértices del grafo.
        Raises: None
        """
        return list(self.adyacencia.keys())

    def lista_adyacencia(self, u: object) -> List[object] or None:
        """Si el objeto u es un vértice del grafo, devuelve
        su lista de adyacencia, es decir, una lista [v1,...,vn] con los vértices
        tales que (u,v1), (u,v2),..., (u,vn) son aristas del grafo.
        Si no, devuelve None.

        Args: u vértice del grafo
        Returns:
            List[object]: Una lista [v1,v2,...,vn] de los vértices del grafo
                adyacentes a u si u es un vértice del grafo
            None: si u no es un vértice del grafo
        Raises:
            TypeError: Si u no es "hashable".
        """
        try:
            if u in self.adyacencia.keys():
                return list(self.adyacencia[u].keys())
            else:
                return None
        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def obtener_aristas(self) -> List[Tuple[object, object, object, float]]:
        """Devuelve todas las aristas del grafo.

        Args: None
        Returns:
            List[Tuple[object, object, object, float]]: Una lista de tuplas, cada una representando
                una arista. Cada tupla contiene el vértice de origen, el vértice de destino,
                los datos asociados a la arista, y su peso.
        Raises: None
        """
        aristas = []
        aristas_vistas = set()  # Conjunto para evitar duplicados
        for u in self.adyacencia:
            for v, (datos, peso) in self.adyacencia[u].items():
                if not self.dirigido:
                    # Utilizar un conjunto para evitar duplicados
                    if (v, u) in aristas_vistas:
                        continue
                    aristas_vistas.add((u, v))
                aristas.append((u, v, datos, peso))
        return aristas

    #### Grados de vértices ####
    def grado_saliente(self, v: object) -> int or None:
        """Si el objeto v es un vértice del grafo, devuelve
        su grado saliente, es decir, el número de aristas que parten de v.
        Si no, devuelve None.

        Args:
            v (object): vértice del grafo
        Returns:
            int: El grado saliente de u si el vértice existe
            None: Si el vértice no existe.
        Raises:
            TypeError: Si u no es "hashable".
        """
        try:
            if v in self.adyacencia.keys():
                return int(len(list(self.adyacencia[v].keys())))
            else:
                return None
        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def grado_entrante(self, v: object) -> int or None:
        """Si el objeto v es un vértice del grafo, devuelve
        su grado entrante, es decir, el número de aristas que llegan a v.
        Si no, devuelve None.

        Args:
            v (object): vértice del grafo
        Returns:
            int: El grado entrante de u si el vértice existe
            None: Si el vértice no existe.
        Raises:
            TypeError: Si v no es "hashable".
        """
        try:
            grado_entrante = 0
            if v in self.adyacencia.keys():
                for clave in self.adyacencia.keys():
                    if v in self.adyacencia[clave].keys():
                        grado_entrante += 1
                return grado_entrante
            else:
                return None
        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    def grado(self, v: object) -> int or None:
        """Si el objeto v es un vértice del grafo, devuelve
        su grado si el grafo no es dirigido y su grado saliente si
        es dirigido.
        Si no pertenece al grafo, devuelve None.

        Args:
            v (object): vértice del grafo
        Returns:
            int: El grado grado o grado saliente de u según corresponda
                si el vértice existe
            None: Si el vértice no existe.
        Raises:
            TypeError: Si v no es "hashable".
        """
        try:
            if v in self.adyacencia.keys():
                if self.dirigido:
                    return self.grado_saliente(v)
                else:
                    return len(list(self.adyacencia[v].keys()))
            else:
                return None

        except TypeError as e:
            raise f"{e} El objeto no es hasheable"

    #### Algoritmos ####
    def dijkstra(self, origen: object) -> Dict[object, object]:
        """Calcula un Árbol de Caminos Mínimos para el grafo partiendo
        del vértice "origen" usando el algoritmo de Dijkstra.

        Args:
            origen (object): vértice del grafo de origen
        Returns:
            Dict[object, object]: Devuelve un diccionario que indica, para cada vértice alcanzable
                desde "origen", qué vértice es su padre en el árbol de caminos mínimos.
        Raises:
            TypeError: Si origen no es "hashable".
        Example:
            Si G.dijksra(1)={2:1, 3:2, 4:1} entonces 1 es padre de 2 y de 4 y 2 es padre de 3.
            En particular, un camino mínimo desde 1 hasta 3 sería 1->2->3.
        """
        distancias = {vertex: INFTY for vertex in self.adyacencia}
        padres = {vertex: None for vertex in self.adyacencia}
        visitados = {vertex: False for vertex in self.adyacencia}

        # Configuración del punto de origen
        distancias[origen] = 0
        cola_prioridad = [(0, origen)]

        while cola_prioridad:
            # Encuentra el nodo con la menor distancia no visitado
            distancia_actual, v_actual = min(
                (d, v) for d, v in cola_prioridad if not visitados[v]
            )

            # Marcar como visitado
            visitados[v_actual] = True

            # Actualizar la cola quitando el nodo visitado
            cola_prioridad = [(d, v) for d, v in cola_prioridad if v != v_actual]

            # Explorar los vecinos del nodo actual
            for vecino, info in self.adyacencia[v_actual].items():
                _, edge_weight = info
                distance = distancia_actual + edge_weight

                # Actualizar la distancia y el padre si se encuentra un camino más corto o igual
                if distance <= distancias[vecino]:
                    distancias[vecino] = distance
                    padres[vecino] = v_actual
                    cola_prioridad.append((distance, vecino))

        # Eliminar el origen del diccionario de padres
        del padres[origen]

        return padres

    def camino_minimo(self, origen: object, destino: object) -> List[object]:
        """Calcula el camino mínimo desde el vértice origen hasta el vértice
        destino utilizando el algoritmo de Dijkstra.

        Args:
            origen (object): vértice del grafo de origen
            destino (object): vértice del grafo de destino
        Returns:
            List[object]: Devuelve una lista con los vértices del grafo por los que pasa
                el camino más corto entre el origen y el destino. El primer elemento de
                la lista es origen y el último destino.
        Example:
            Si G.dijksra(1,4)=[1,5,2,4] entonces el camino más corto en G entre 1 y 4 es 1->5->2->4.
        Raises:
            TypeError: Si origen o destino no son "hashable".
        """
        try:
            hash(origen)
            hash(destino)
            padres = self.dijkstra(origen)
            camino = [destino]

            while origen != destino:
                camino.append(padres[destino])
                destino = padres[destino]

            return camino[::-1]
        except ValueError as e:
            print(f"{e}, el origen o el destino no es hasheable")

    def prim(self) -> Dict[object, object]:
        """Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Prim.

        Args: None
        Returns:
            Dict[object,object]: Devuelve un diccionario que indica, para cada vértice del
                grafo, qué vértice es su padre en el árbol abarcador mínimo.
        Raises: None
        Example:
            Si G.prim()={2:1, 3:2, 4:1} entonces en un árbol abarcador mínimo tenemos que:
                1 es padre de 2 y de 4
                2 es padre de 3
        Eugenio
        """
        try:
            # Inicializar el conjunto de vértices visitados
            visited = set()

            # Seleccionar un nodo inicial (puedes elegir cualquier nodo)
            start_node = list(self.adyacencia.keys())[0]

            # Inicializar la cola de prioridad (heap) con las conexiones del nodo inicial
            heap = [
                (weight, start_node, neighbor)
                for neighbor, (data, weight) in self.adyacencia[start_node].items()
            ]
            heapq.heapify(heap)

            # Estructura para almacenar el árbol de expansión mínima
            minimum_spanning_tree = {start_node: None}

            while heap:
                # Obtener la conexión con el peso mínimo
                weight, current_node, next_node = heapq.heappop(heap)

                # Si el siguiente nodo no ha sido visitado, agregarlo al árbol
                if next_node not in visited:
                    visited.add(next_node)
                    minimum_spanning_tree[next_node] = current_node

                    # Agregar las conexiones del próximo nodo a la cola de prioridad
                    for neighbor, (data, w) in self.adyacencia[next_node].items():
                        if neighbor not in visited:
                            heapq.heappush(heap, (w, next_node, neighbor))

            return dict(sorted(minimum_spanning_tree.items()))

        except ValueError as e:
            print(f"{e}, Alguno de los vértices no es hasheable")

    def kruskal(self) -> List[Tuple[object, object]]:
        """Calcula un Árbol Abarcador Mínimo para el grafo
        usando el algoritmo de Kruskal.

        Args: None
        Returns:
            List[Tuple[object,object]]: Devuelve una lista [(s1,t1),(s2,t2),...,(sn,tn)]
                de los pares de vértices del grafo que forman las aristas
                del arbol abarcador mínimo.
        Raises: None
        Example:
            En el ejemplo anterior en que G.kruskal()={2:1, 3:2, 4:1} podríamos tener, por ejemplo,
            G.prim=[(1,2),(1,4),(3,2)]
        Eugenio
        """
        try:
            # Inicializar el conjunto de vértices visitados
            visited = set()

            # Seleccionar un nodo inicial (puedes elegir cualquier nodo)
            start_node = list(self.adyacencia.keys())[0]

            # Inicializar la cola de prioridad (heap) con las conexiones del nodo inicial
            heap = [
                (weight, start_node, neighbor)
                for neighbor, (data, weight) in self.adyacencia[start_node].items()
            ]
            heapq.heapify(heap)

            # Estructura para almacenar el árbol de expansión mínima
            tree_list = []

            while heap:
                # Obtener la conexión con el peso mínimo
                weight, current_node, next_node = heapq.heappop(heap)

                # Si el siguiente nodo no ha sido visitado, agregarlo al árbol
                if next_node not in visited:
                    visited.add(next_node)
                    tree_list.append((next_node, current_node))

                    # Agregar las conexiones del próximo nodo a la cola de prioridad
                    for neighbor, (data, w) in self.adyacencia[next_node].items():
                        if neighbor not in visited:
                            heapq.heappush(heap, (w, next_node, neighbor))

            lista_ordenada = sorted(tree_list, key=lambda x: x[0])
            return lista_ordenada

        except ValueError as e:
            print(f"{e}, Alguno de los vértices no es hasheable")

    #### NetworkX ####
    def convertir_a_NetworkX(self) -> nx.Graph or nx.DiGraph:
        """Construye un grafo o digrafo de Networkx según corresponda
        a partir de los datos del grafo actual.

        Args: None
        Returns:
            networkx.Graph: Objeto Graph de NetworkX si el grafo es no dirigido.
            networkx.DiGraph: Objeto DiGraph si el grafo es dirigido.
            En ambos casos, los vértices y las aristas son los contenidos en el grafo dado.
        Raises: None
        Eugenio
        """
        tipo_grafo = nx.DiGraph if self.dirigido else nx.Graph
        grafo = tipo_grafo()

        nodos = list(self.adyacencia.keys())
        grafo.add_nodes_from(nodos)

        lista_adyacencias = []
        for v in nodos:
            for ady in self.adyacencia[v].keys():
                lista_adyacencias.append(
                    (
                        v,
                        ady,
                        {
                            "data": self.adyacencia[v][ady][0],
                            "weight": self.adyacencia[v][ady][1],
                        },
                    )
                )

        grafo.add_edges_from(lista_adyacencias)

        return grafo
