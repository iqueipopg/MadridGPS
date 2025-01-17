"""
navegador.py

Matemática Discreta - IMAT
ICAI, Universidad Pontificia Comillas

Grupo: GP13B
Integrantes:
    - Eugenio Ribón Novoa
    - Ignacio Queipo de Llano Pérez-Gascón

Descripción:
Librería para la creación y análisis de la aplicación GPS.
"""

from navegador import *


calle1_prueba = "GONZALEZ DAVILA,NUM000001"
calle2_prueba = "FEDERICO MORENO TORROBA,NUM000001 A"
print(f"Procesando información y creando grafos...")
grafo_dist, grafo_tiempo, calles, df_cruces, df_direcciones = crear_grafos(
    "data/cruces.csv", "data/direcciones.csv"
)

salir = False

while not salir:
    opc = int(input("Desea buscar una ruta[0] o salir del navegador[1]? "))
    if opc == 1:
        salir = True
    elif opc == 0:
        calle1 = input(
            """Introduzca el origen con el siguiente formato:
                            Calles   -> NOMBRE DE LA VIA,NUMXXXXXX
                            Autovias -> NOMBRE DE LA VIA,KM.XXXXXXEN/SA
                        """
        )
        calle2 = input(
            """Introduzca el destino con el siguiente formato:
                            Calles   -> NOMBRE DE LA VIA,NUMXXXXXX
                            Autovias -> NOMBRE DE LA VIA,KM.XXXXXXEN/SA
                        """
        )

        calle1_obj, literal1 = calle_to_obj(calle1, calles)
        calle2_obj, literal2 = calle_to_obj(calle2, calles)
        if calle1_obj and calle2_obj:
            cruce_inicio, inicio = calle_cerca_cruce(calle1_obj, literal1, inicio=True)
            cruce_final, final = calle_cerca_cruce(calle2_obj, literal2)
            eligiendo_forma = True
            while eligiendo_forma:
                forma_llegar = input(
                    "Seleccione como calcular la ruta: [D]istancia o [T]iempo "
                )

                if forma_llegar == "D":
                    eligiendo_forma = False
                    ruta = ruta_minima(grafo_dist, cruce_inicio, cruce_final)
                    ins = instrucciones(ruta, inicio, final)
                    mostrar_instr(ins)

                    grafo_nx = grafo_dist.convertir_a_NetworkX()
                    posiciones = {
                        cruce: (cruce.coord_x, cruce.coord_y)
                        for cruce in grafo_nx.nodes()
                    }
                    plot = plt.plot()

                    nx.draw_networkx(
                        grafo_nx,
                        pos=posiciones,
                        with_labels=False,
                        node_size=0.1,
                        width=0.1,
                    )

                    camino_a_resaltar = ruta

                    # Resaltar el camino pintando las aristas en rojo
                    edges_to_highlight = [
                        (camino_a_resaltar[i], camino_a_resaltar[i + 1])
                        for i in range(len(camino_a_resaltar) - 1)
                    ]
                    nx.draw_networkx_edges(
                        grafo_nx,
                        pos=posiciones,
                        edgelist=edges_to_highlight,
                        edge_color="red",
                        width=2.0,
                    )

                    plt.show()

                elif forma_llegar == "T":
                    eligiendo_forma = False
                    ruta = ruta_minima(grafo_tiempo, cruce_inicio, cruce_final)
                    ins = instrucciones(ruta, inicio, final)
                    mostrar_instr(ins)

                    grafo_nx = grafo_dist.convertir_a_NetworkX()
                    posiciones = {
                        cruce: (cruce.coord_x, cruce.coord_y)
                        for cruce in grafo_nx.nodes()
                    }
                    plot = plt.plot()

                    nx.draw_networkx(
                        grafo_nx,
                        pos=posiciones,
                        with_labels=False,
                        node_size=0.1,
                        width=0.1,
                    )

                    camino_a_resaltar = ruta

                    # Resaltar el camino pintando las aristas en rojo
                    edges_to_highlight = [
                        (camino_a_resaltar[i], camino_a_resaltar[i + 1])
                        for i in range(len(camino_a_resaltar) - 1)
                    ]
                    nx.draw_networkx_edges(
                        grafo_nx,
                        pos=posiciones,
                        edgelist=edges_to_highlight,
                        edge_color="red",
                        width=2.0,
                    )

                    plt.show()

    else:
        print("Porfavor eliga una opción válida")
