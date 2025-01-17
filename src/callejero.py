"""
callejero.py

Matemática Discreta - IMAT
ICAI, Universidad Pontificia Comillas

Grupo: GP13B
Integrantes:
    - XX
    - XX

Descripción:
Librería con herramientas y clases auxiliares necesarias para la representación de un callejero en un grafo.
"""

# Constantes con las velocidades máximas establecidas por el enunciado para cada tipo de vía.
VELOCIDADES_CALLES = {
    "AUTOVIA": 100,
    "AVENIDA": 90,
    "CARRETERA": 70,
    "CALLEJON": 30,
    "CAMINO": 30,
    "ESTACION DE METRO": 20,
    "PASADIZO": 20,
    "PLAZUELA": 20,
    "COLONIA": 20,
}
VELOCIDAD_CALLES_ESTANDAR = 50


class Cruce:
    # Completar esta clase con los datos y métodos que se necesite asociar a cada cruce

    def __init__(self, coord_x: int, coord_y: int):
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.codigo_calles_cruce = []
        self.dirs_calles_cruce = []
        self.objetos_cruces = []
        self.nombre_calle = None
        # Completar la inicialización de las estructuras de datos agregadas

    """Se hace que la clase Cruce sea "hashable" mediante la implementación de los métodos
    __eq__ y __hash__, haciendo que dos objetos de tipo Cruce se consideren iguales cuando
    sus coordenadas coincidan (es decir, C1==C2 si y sólo si C1 y C2 tienen las mismas coordenadas),
    independientemente de los otros campos que puedan estar almacenados en los objetos.
    La función __hash__ se adapta en consecuencia para que sólo dependa del par (coord_x, coord_y).
    """

    def agregar_dir_calle(self, dir: str):
        self.dirs_calles_cruce.append(dir)

    def agregar_cod_calle(self, cod: int):
        self.codigo_calles_cruce.append(cod)

    """Se hace que la clase Cruce sea "hashable" mediante la implementación de los métodos
    __eq__ y __hash__, haciendo que dos objetos de tipo Cruce se consideren iguales cuando
    sus coordenadas coincidan (es decir, C1==C2 si y sólo si C1 y C2 tienen las mismas coordenadas),
    independientemente de los otros campos que puedan estar almacenados en los objetos.
    La función __hash__ se adapta en consecuencia para que sólo dependa del par (coord_x, coord_y).
    """

    def __eq__(self, other) -> int:
        if type(other) is type(self):
            return (self.coord_x == other.coord_x) and (self.coord_y == other.coord_y)
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.coord_x, self.coord_y))

    def __str__(self) -> str:
        return f"Cruce en ({self.coord_x}, {self.coord_y}"


class Calle:
    # Completar esta clase con los datos que sea necesario almacenar de cada calle para poder reconstruir los datos del callejero
    def __init__(self):
        self.velocidad = set()
        self.dirs = []
        self.codigo = None
        self.nombre = None
        self.coordx = []
        self.coordy = []
        self.tipo_calle = None
        self.literal_numeracion_calle = []
        self.objetos_cruces = []

    def __str__(self):
        return f""""
        Calle {self.nombre}:
            Codigo: {self.codigo}
            Tipo de la vía: {self.tipo_calle}
            Coordenadas: ({self.coordx}, {self.coordy})
            Velocidad: {self.velocidad} km/h
            """

    def agg_velocidad(self):
        tipo_calle = self.tipo_calle.upper().strip()
        try:
            if tipo_calle in VELOCIDADES_CALLES.keys():
                return int(VELOCIDADES_CALLES[tipo_calle])
            else:
                return 50
        except ValueError:
            print("Error: no se ha podido obtener la velocidad de la calle.")
            print(f"Código de la calle: {self.tipo_calle}")
